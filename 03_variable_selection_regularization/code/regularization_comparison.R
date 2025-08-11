# Regularization Comparison: Ridge vs Lasso
# This script demonstrates ridge and lasso regression using the glmnet package

# Load required libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Set random seed for reproducibility
set.seed(42)

# Generate synthetic data
n <- 100
p <- 20
X <- matrix(rnorm(n * p), n, p)
true_beta <- rep(0, p)
true_beta[1:5] <- c(2, -1.5, 1, -0.8, 0.6)
y <- X %*% true_beta + 0.5 * rnorm(n)

# Split data into training and test sets
train_idx <- sample(1:n, 0.7 * n)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

cat("=== REGULARIZATION COMPARISON: RIDGE VS LASSO ===\n")
cat("Sample size:", n, "\n")
cat("Number of predictors:", p, "\n")
cat("Training set size:", length(train_idx), "\n")
cat("Test set size:", n - length(train_idx), "\n")
cat("True non-zero coefficients:", sum(true_beta != 0), "\n\n")

# Fit ridge regression with cross-validation
cat("=== RIDGE REGRESSION ===\n")
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, standardize = TRUE)
ridge_fit <- glmnet(X_train, y_train, alpha = 0, lambda = ridge_cv$lambda.min)

cat("Best ridge lambda:", round(ridge_cv$lambda.min, 4), "\n")
cat("Ridge CV MSE:", round(min(ridge_cv$cvm), 4), "\n")

# Fit lasso regression with cross-validation
cat("\n=== LASSO REGRESSION ===\n")
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE)
lasso_fit <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min)

cat("Best lasso lambda:", round(lasso_cv$lambda.min, 4), "\n")
cat("Lasso CV MSE:", round(min(lasso_cv$cvm), 4), "\n")

# Plot coefficient paths
par(mfrow = c(1, 2))

# Ridge coefficient paths
plot(ridge_cv$glmnet.fit, xvar = "lambda", main = "Ridge: Coefficient Paths")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = 2)

# Lasso coefficient paths
plot(lasso_cv$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = 2)

# Make predictions on test set
ridge_pred <- predict(ridge_fit, newx = X_test)
lasso_pred <- predict(lasso_fit, newx = X_test)

# Calculate R-squared for both models
ridge_r2 <- 1 - sum((y_test - ridge_pred)^2) / sum((y_test - mean(y_test))^2)
lasso_r2 <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)

# Calculate MSE for both models
ridge_mse <- mean((y_test - ridge_pred)^2)
lasso_mse <- mean((y_test - lasso_pred)^2)

# Model comparison results
cat("\n=== MODEL COMPARISON ===\n")
cat("Ridge R²:", round(ridge_r2, 4), "\n")
cat("Lasso R²:", round(lasso_r2, 4), "\n")
cat("Ridge MSE:", round(ridge_mse, 4), "\n")
cat("Lasso MSE:", round(lasso_mse, 4), "\n")
cat("Ridge non-zero coefficients:", sum(coef(ridge_fit) != 0), "\n")
cat("Lasso non-zero coefficients:", sum(coef(lasso_fit) != 0), "\n")

# Extract coefficients for comparison
ridge_coef <- as.vector(coef(ridge_fit))
lasso_coef <- as.vector(coef(lasso_fit))

# Create coefficient comparison plot
coef_comparison <- data.frame(
  Variable = 0:p,
  Ridge = ridge_coef,
  Lasso = lasso_coef,
  True = c(0, true_beta)  # Include intercept
)

# Plot coefficient comparison
ggplot(coef_comparison, aes(x = Variable)) +
  geom_line(aes(y = Ridge, color = "Ridge"), size = 1) +
  geom_line(aes(y = Lasso, color = "Lasso"), size = 1) +
  geom_point(aes(y = True, color = "True"), size = 2) +
  scale_color_manual(values = c("Ridge" = "blue", "Lasso" = "red", "True" = "green")) +
  labs(title = "Coefficient Comparison: Ridge vs Lasso vs True",
       x = "Variable Index",
       y = "Coefficient Value",
       color = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print coefficient summary
cat("\n=== COEFFICIENT SUMMARY ===\n")
cat("True non-zero coefficients (indices):", which(true_beta != 0), "\n")
cat("Ridge non-zero coefficients (indices):", which(ridge_coef[-1] != 0), "\n")
cat("Lasso non-zero coefficients (indices):", which(lasso_coef[-1] != 0), "\n")

# Calculate coefficient accuracy
ridge_accuracy <- sum((ridge_coef[-1] != 0) == (true_beta != 0)) / p
lasso_accuracy <- sum((lasso_coef[-1] != 0) == (true_beta != 0)) / p

cat("\n=== VARIABLE SELECTION ACCURACY ===\n")
cat("Ridge variable selection accuracy:", round(ridge_accuracy, 4), "\n")
cat("Lasso variable selection accuracy:", round(lasso_accuracy, 4), "\n")

# Key insights
cat("\n=== KEY INSIGHTS ===\n")
cat("1. Ridge regression keeps all variables but shrinks coefficients\n")
cat("2. Lasso regression performs automatic variable selection\n")
cat("3. Cross-validation helps select optimal regularization parameters\n")
cat("4. Lasso provides sparsity while ridge provides stability\n")
cat("5. Both methods can improve prediction accuracy over OLS\n")

# Return results for further analysis
results <- list(
  ridge_cv = ridge_cv,
  lasso_cv = lasso_cv,
  ridge_fit = ridge_fit,
  lasso_fit = lasso_fit,
  ridge_coef = ridge_coef,
  lasso_coef = lasso_coef,
  ridge_r2 = ridge_r2,
  lasso_r2 = lasso_r2,
  ridge_mse = ridge_mse,
  lasso_mse = lasso_mse,
  true_beta = true_beta
)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results stored in 'results' object for further analysis\n")
