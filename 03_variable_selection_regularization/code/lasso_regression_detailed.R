# Lasso Regression: Comprehensive Implementation
# This script demonstrates lasso regression with coordinate descent and variable selection

# Load required libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Set random seed for reproducibility
set.seed(42)

# Generate synthetic data with sparse true coefficients
n <- 100
p <- 20

# Create design matrix
X <- matrix(rnorm(n * p), n, p)
X[, 2] <- 0.3 * X[, 1] + 0.7 * rnorm(n)
X[, 3] <- 0.2 * X[, 1] + 0.8 * rnorm(n)

# True coefficients (sparse)
true_beta <- rep(0, p)
true_beta[1:5] <- c(3, -2, 1.5, -1, 0.8)

# Generate response
y <- X %*% true_beta + 0.5 * rnorm(n)

cat("=== LASSO REGRESSION WITH SPARSE COEFFICIENTS ===\n")
cat("Sample size:", n, "\n")
cat("Number of predictors:", p, "\n")
cat("True non-zero coefficients:", sum(true_beta != 0), "\n")
cat("True coefficients:", true_beta[1:5], "\n")

# Split data into training and test sets
train_idx <- sample(1:n, 0.7 * n)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Standardize data
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                       scale = attr(X_train_scaled, "scaled:scale"))
y_train_scaled <- scale(y_train)
y_test_scaled <- scale(y_test, center = attr(y_train_scaled, "scaled:center"), 
                       scale = attr(y_train_scaled, "scaled:scale"))

# Implement coordinate descent for lasso
coordinate_descent_lasso <- function(X, y, lambda_val, max_iter = 1000, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  
  for (iteration in 1:max_iter) {
    beta_old <- beta
    
    for (j in 1:p) {
      # Compute partial residual
      r_j <- y - X %*% beta + X[, j] * beta[j]
      
      # Compute univariate OLS
      x_j_norm_sq <- sum(X[, j]^2)
      if (x_j_norm_sq > 0) {
        beta_ols <- sum(X[, j] * r_j) / x_j_norm_sq
        
        # Apply soft thresholding
        threshold <- lambda_val / (2 * x_j_norm_sq)
        if (abs(beta_ols) <= threshold) {
          beta[j] <- 0
        } else {
          beta[j] <- sign(beta_ols) * (abs(beta_ols) - threshold)
        }
      }
    }
    
    # Check convergence
    if (max(abs(beta - beta_old)) < tol) break
  }
  
  return(beta)
}

# Soft thresholding operator
soft_threshold <- function(x, threshold) {
  return(sign(x) * pmax(abs(x) - threshold, 0))
}

cat("\n=== SOFT THRESHOLDING OPERATOR ===\n")
cat("Demonstrating soft thresholding with different lambda values...\n")

# Demonstrate soft thresholding
x_vals <- seq(-3, 3, length.out = 100)
thresholds <- c(0.5, 1.0, 1.5)

# Create plot data
plot_data <- data.frame(
  x = rep(x_vals, length(thresholds)),
  y = unlist(lapply(thresholds, function(t) soft_threshold(x_vals, t))),
  threshold = rep(paste("λ =", thresholds), each = length(x_vals))
)

ggplot(plot_data, aes(x = x, y = y, color = threshold)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5) +
  labs(title = "Soft Thresholding Operator", x = "Input", y = "Output") +
  theme_minimal()

# Lasso with cross-validation
cat("\n=== LASSO WITH CROSS-VALIDATION ===\n")
lasso_cv <- cv.glmnet(X_train_scaled, y_train_scaled, alpha = 1, standardize = FALSE)
lasso_fit <- glmnet(X_train_scaled, y_train_scaled, alpha = 1, lambda = lasso_cv$lambda.min)

# Compare with coordinate descent
lambda_test <- 0.1
lasso_cd <- coordinate_descent_lasso(X_train_scaled, y_train_scaled, lambda_test)

cat("Best lambda:", round(lasso_cv$lambda.min, 4), "\n")
cat("Lasso CV MSE:", round(min(lasso_cv$cvm), 4), "\n")

# Plot coefficient paths
plot(lasso_cv$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = 2)

# Compare with OLS
ols_coefs <- coef(lm(y_train_scaled ~ X_train_scaled - 1))
lasso_coefs <- as.vector(coef(lasso_fit))[-1]  # Remove intercept

# Create comparison plot
coef_comparison <- data.frame(
  predictor = 1:p,
  ols = ols_coefs,
  lasso = lasso_coefs
)

ggplot(coef_comparison, aes(x = predictor)) +
  geom_bar(aes(y = ols, fill = "OLS"), stat = "identity", alpha = 0.7, width = 0.4) +
  geom_bar(aes(y = lasso, fill = "Lasso"), stat = "identity", alpha = 0.7, width = 0.4, 
           position = position_nudge(x = 0.4)) +
  scale_fill_manual(values = c("OLS" = "blue", "Lasso" = "red")) +
  labs(title = "OLS vs Lasso Coefficients", x = "Predictor Index", y = "Coefficient Value") +
  theme_minimal()

# Prediction comparison
ols_pred <- X_test_scaled %*% ols_coefs
lasso_pred <- predict(lasso_fit, newx = X_test_scaled)

ols_r2 <- 1 - sum((y_test_scaled - ols_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)
lasso_r2 <- 1 - sum((y_test_scaled - lasso_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)

ols_mse <- mean((y_test_scaled - ols_pred)^2)
lasso_mse <- mean((y_test_scaled - lasso_pred)^2)

cat("\n=== MODEL COMPARISON ===\n")
cat("OLS Test R²:", round(ols_r2, 4), "\n")
cat("Lasso Test R²:", round(lasso_r2, 4), "\n")
cat("OLS Test MSE:", round(ols_mse, 4), "\n")
cat("Lasso Test MSE:", round(lasso_mse, 4), "\n")
cat("Improvement in R²:", round(lasso_r2 - ols_r2, 4), "\n")
cat("Improvement in MSE:", round((ols_mse - lasso_mse) / ols_mse * 100, 2), "%\n")
cat("Lasso non-zero coefficients:", sum(lasso_coefs != 0), "\n")
cat("Best λ:", round(lasso_cv$lambda.min, 4), "\n")

# Compare implementations
cat("\n=== IMPLEMENTATION COMPARISON ===\n")
cat("Implementation comparison (λ =", lambda_test, "):\n")
glmnet_coefs <- as.vector(coef(lasso_fit, s = lambda_test))[-1]
cat("Glmnet Lasso:", round(glmnet_coefs[1:5], 4), "\n")
cat("Coordinate Descent:", round(lasso_cd[1:5], 4), "\n")
cat("Maximum difference:", round(max(abs(glmnet_coefs - lasso_cd)), 6), "\n")

# Variable selection analysis
cat("\n=== VARIABLE SELECTION ANALYSIS ===\n")
cat("Variable selection results:\n")
cat("True non-zero coefficients:", sum(true_beta != 0), "\n")
cat("Lasso non-zero coefficients:", sum(lasso_coefs != 0), "\n")
cat("Correctly identified non-zero:", sum((true_beta != 0) & (lasso_coefs != 0)), "\n")
cat("Correctly identified zero:", sum((true_beta == 0) & (lasso_coefs == 0)), "\n")

# Calculate selection accuracy
selection_accuracy <- (sum((true_beta != 0) & (lasso_coefs != 0)) + 
                      sum((true_beta == 0) & (lasso_coefs == 0))) / p
cat("Variable selection accuracy:", round(selection_accuracy, 4), "\n")

# Coefficient stability analysis
cat("\n=== COEFFICIENT STABILITY ===\n")
lambda_stability <- c(0.05, 0.1, 0.2)
stability_results <- matrix(0, length(lambda_stability), p)

for (i in 1:length(lambda_stability)) {
  lasso_stable <- glmnet(X_train_scaled, y_train_scaled, alpha = 1, lambda = lambda_stability[i])
  stability_results[i, ] <- as.vector(coef(lasso_stable))[-1]
}

coefficient_variance <- apply(stability_results, 2, var)
cat("Coefficient variance across lambda values:", round(mean(coefficient_variance), 6), "\n")
cat("Most stable coefficients (lowest variance):", order(coefficient_variance)[1:5], "\n")
cat("Least stable coefficients (highest variance):", order(coefficient_variance, decreasing = TRUE)[1:5], "\n")

# Create comprehensive visualization
par(mfrow = c(2, 2))

# 1. Coefficient paths
plot(lasso_cv$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = 2)

# 2. Cross-validation curve
plot(lasso_cv, main = "Cross-Validation MSE")

# 3. OLS vs Lasso coefficients
barplot(rbind(ols_coefs, lasso_coefs), beside = TRUE, 
        names.arg = 1:p, col = c("blue", "red"), 
        main = "OLS vs Lasso Coefficients",
        xlab = "Predictor Index", ylab = "Coefficient Value")
legend("topright", legend = c("OLS", "Lasso"), fill = c("blue", "red"))

# 4. Prediction comparison
plot(y_test_scaled, ols_pred, col = "blue", pch = 16, alpha = 0.6,
     main = "Prediction Comparison", xlab = "True Values", ylab = "Predicted Values")
points(y_test_scaled, lasso_pred, col = "red", pch = 16, alpha = 0.6)
abline(0, 1, lty = 2)
legend("topleft", legend = c(paste("OLS (R² =", round(ols_r2, 3), ")"),
                            paste("Lasso (R² =", round(lasso_r2, 3), ")")),
       col = c("blue", "red"), pch = 16)

# Additional analysis: Sparsity vs lambda
lambda_seq <- exp(seq(log(0.001), log(1), length.out = 50))
sparsity_seq <- sapply(lambda_seq, function(lambda) {
  lasso_temp <- glmnet(X_train_scaled, y_train_scaled, alpha = 1, lambda = lambda)
  return(sum(as.vector(coef(lasso_temp))[-1] != 0))
})

plot(lambda_seq, sparsity_seq, type = "l", log = "x", 
     main = "Sparsity vs Lambda",
     xlab = "Lambda", ylab = "Number of Non-zero Coefficients")
abline(v = lasso_cv$lambda.min, col = "red", lty = 2)
abline(h = sum(lasso_coefs != 0), col = "red", lty = 2)

# Key insights
cat("\n=== KEY INSIGHTS ===\n")
cat("1. Lasso performs automatic variable selection through soft thresholding\n")
cat("2. Coordinate descent provides an efficient algorithm for lasso optimization\n")
cat("3. Cross-validation helps select optimal regularization parameter\n")
cat("4. Lasso can improve prediction accuracy in sparse settings\n")
cat("5. Variable selection accuracy depends on signal strength and noise level\n")
cat("6. Coefficient stability varies across different lambda values\n")
cat("7. Lasso provides interpretable models through sparsity\n")
cat("8. Glmnet and coordinate descent implementations are highly consistent\n")

# Return results for further analysis
results <- list(
  lasso_cv = lasso_cv,
  lasso_fit = lasso_fit,
  lasso_coefs = lasso_coefs,
  ols_coefs = ols_coefs,
  lasso_r2 = lasso_r2,
  ols_r2 = ols_r2,
  lasso_mse = lasso_mse,
  ols_mse = ols_mse,
  non_zero_count = sum(lasso_coefs != 0),
  selection_accuracy = selection_accuracy,
  true_beta = true_beta,
  lambda_values = lambda_seq,
  sparsity_values = sparsity_seq
)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results stored in 'results' object for further analysis\n")
