# Ridge Regression: Comprehensive Implementation
# This script demonstrates ridge regression with multicollinearity handling using glmnet

# Load required libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Set random seed for reproducibility
set.seed(42)

# Generate synthetic data with multicollinearity
n <- 100
p <- 10

# Create correlated predictors
X <- matrix(rnorm(n * p), n, p)
X[, 2] <- 0.8 * X[, 1] + 0.2 * rnorm(n)
X[, 3] <- 0.7 * X[, 1] + 0.3 * rnorm(n)

# True coefficients
true_beta <- rep(0, p)
true_beta[1:3] <- c(2, -1.5, 1)

# Generate response
y <- X %*% true_beta + 0.5 * rnorm(n)

cat("=== RIDGE REGRESSION WITH MULTICOLLINEARITY ===\n")
cat("Sample size:", n, "\n")
cat("Number of predictors:", p, "\n")
cat("True non-zero coefficients:", sum(true_beta != 0), "\n")
cat("True coefficients:", true_beta[1:3], "\n")

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

# Compute SVD
svd_result <- svd(X_train_scaled)
U <- svd_result$u
d <- svd_result$d
V <- svd_result$v

cat("\nSVD Analysis:\n")
cat("Singular values:", round(d[1:5], 3), "...\n")
cat("Condition number:", round(d[1] / d[length(d)], 2), "\n")

# Ridge regression with cross-validation
cat("\n=== RIDGE REGRESSION WITH CROSS-VALIDATION ===\n")
ridge_cv <- cv.glmnet(X_train_scaled, y_train_scaled, alpha = 0, standardize = FALSE)
ridge_fit <- glmnet(X_train_scaled, y_train_scaled, alpha = 0, lambda = ridge_cv$lambda.min)

# Compute degrees of freedom
df_ridge <- sum(d^2 / (d^2 + ridge_cv$lambda.min))

cat("Best lambda:", round(ridge_cv$lambda.min, 4), "\n")
cat("Ridge CV MSE:", round(min(ridge_cv$cvm), 4), "\n")
cat("Degrees of freedom:", round(df_ridge, 2), "\n")

# Plot coefficient paths
plot(ridge_cv$glmnet.fit, xvar = "lambda", main = "Ridge: Coefficient Paths")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = 2)

# Compare with OLS
ols_coefs <- coef(lm(y_train_scaled ~ X_train_scaled - 1))
ridge_coefs <- as.vector(coef(ridge_fit))[-1]  # Remove intercept

# Create comparison plot
coef_comparison <- data.frame(
  predictor = 1:p,
  ols = ols_coefs,
  ridge = ridge_coefs
)

ggplot(coef_comparison, aes(x = predictor)) +
  geom_bar(aes(y = ols, fill = "OLS"), stat = "identity", alpha = 0.7, width = 0.4) +
  geom_bar(aes(y = ridge, fill = "Ridge"), stat = "identity", alpha = 0.7, width = 0.4, 
           position = position_nudge(x = 0.4)) +
  scale_fill_manual(values = c("OLS" = "blue", "Ridge" = "red")) +
  labs(title = "OLS vs Ridge Coefficients", x = "Predictor Index", y = "Coefficient Value") +
  theme_minimal()

# Prediction comparison
ols_pred <- X_test_scaled %*% ols_coefs
ridge_pred <- predict(ridge_fit, newx = X_test_scaled)

ols_r2 <- 1 - sum((y_test_scaled - ols_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)
ridge_r2 <- 1 - sum((y_test_scaled - ridge_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)

ols_mse <- mean((y_test_scaled - ols_pred)^2)
ridge_mse <- mean((y_test_scaled - ridge_pred)^2)

cat("\n=== MODEL COMPARISON ===\n")
cat("OLS Test R²:", round(ols_r2, 4), "\n")
cat("Ridge Test R²:", round(ridge_r2, 4), "\n")
cat("OLS Test MSE:", round(ols_mse, 4), "\n")
cat("Ridge Test MSE:", round(ridge_mse, 4), "\n")
cat("Improvement in R²:", round(ridge_r2 - ols_r2, 4), "\n")
cat("Improvement in MSE:", round((ols_mse - ridge_mse) / ols_mse * 100, 2), "%\n")

# Multicollinearity analysis
cat("\n=== MULTICOLLINEARITY ANALYSIS ===\n")
corr_matrix <- cor(X_train_scaled)
max_corr <- max(abs(corr_matrix[upper.tri(corr_matrix)]))
cat("Maximum correlation between predictors:", round(max_corr, 4), "\n")

# Variance Inflation Factor (VIF) calculation
vif_values <- numeric(p)
for (i in 1:p) {
  # Regress predictor i on all other predictors
  X_others <- X_train_scaled[, -i, drop = FALSE]
  y_pred <- X_train_scaled[, i]
  
  # Fit regression
  lm_result <- lm(y_pred ~ X_others - 1)
  r_squared <- summary(lm_result)$r.squared
  
  # Calculate VIF
  vif_values[i] <- 1 / (1 - r_squared)
}

cat("VIF values:", round(vif_values[1:5], 2), "...\n")
cat("Maximum VIF:", round(max(vif_values), 2), "\n")

# Shrinkage analysis
cat("\n=== SHRINKAGE ANALYSIS ===\n")
shrinkage_factors <- d^2 / (d^2 + ridge_cv$lambda.min)
cat("Shrinkage factors for first 5 components:", round(shrinkage_factors[1:5], 3), "\n")
cat("Average shrinkage factor:", round(mean(shrinkage_factors), 3), "\n")

# Coefficient stability analysis
cat("\n=== COEFFICIENT STABILITY ===\n")
coef_stability <- abs(ridge_coefs) / abs(ols_coefs)
cat("Coefficient stability ratios:", round(coef_stability[1:5], 3), "...\n")
cat("Average stability ratio:", round(mean(coef_stability), 3), "\n")

# Create comprehensive visualization
par(mfrow = c(2, 2))

# 1. Coefficient paths
plot(ridge_cv$glmnet.fit, xvar = "lambda", main = "Ridge: Coefficient Paths")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = 2)

# 2. Cross-validation curve
plot(ridge_cv, main = "Cross-Validation MSE")

# 3. OLS vs Ridge coefficients
barplot(rbind(ols_coefs, ridge_coefs), beside = TRUE, 
        names.arg = 1:p, col = c("blue", "red"), 
        main = "OLS vs Ridge Coefficients",
        xlab = "Predictor Index", ylab = "Coefficient Value")
legend("topright", legend = c("OLS", "Ridge"), fill = c("blue", "red"))

# 4. Prediction comparison
plot(y_test_scaled, ols_pred, col = "blue", pch = 16, alpha = 0.6,
     main = "Prediction Comparison", xlab = "True Values", ylab = "Predicted Values")
points(y_test_scaled, ridge_pred, col = "red", pch = 16, alpha = 0.6)
abline(0, 1, lty = 2)
legend("topleft", legend = c(paste("OLS (R² =", round(ols_r2, 3), ")"),
                            paste("Ridge (R² =", round(ridge_r2, 3), ")")),
       col = c("blue", "red"), pch = 16)

# Additional analysis: Degrees of freedom vs lambda
lambda_seq <- exp(seq(log(0.001), log(100), length.out = 50))
df_seq <- sapply(lambda_seq, function(lambda) sum(d^2 / (d^2 + lambda)))

plot(lambda_seq, df_seq, type = "l", log = "x", 
     main = "Degrees of Freedom vs Lambda",
     xlab = "Lambda", ylab = "Degrees of Freedom")
abline(v = ridge_cv$lambda.min, col = "red", lty = 2)
abline(h = df_ridge, col = "red", lty = 2)

# Key insights
cat("\n=== KEY INSIGHTS ===\n")
cat("1. Ridge regression handles multicollinearity effectively\n")
cat("2. Cross-validation helps select optimal regularization parameter\n")
cat("3. Ridge shrinks coefficients but doesn't set them to zero\n")
cat("4. Degrees of freedom decrease with increasing regularization\n")
cat("5. Ridge can improve prediction accuracy in presence of multicollinearity\n")
cat("6. VIF analysis confirms presence of multicollinearity\n")
cat("7. Shrinkage factors show differential regularization by component\n")

# Return results for further analysis
results <- list(
  ridge_cv = ridge_cv,
  ridge_fit = ridge_fit,
  ridge_coefs = ridge_coefs,
  ols_coefs = ols_coefs,
  ridge_r2 = ridge_r2,
  ols_r2 = ols_r2,
  ridge_mse = ridge_mse,
  ols_mse = ols_mse,
  df_ridge = df_ridge,
  singular_values = d,
  vif_values = vif_values,
  shrinkage_factors = shrinkage_factors,
  true_beta = true_beta
)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results stored in 'results' object for further analysis\n")
