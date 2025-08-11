# Polynomial Regression Analysis
# This file contains R code for polynomial regression analysis
# Extracted from Rcode_W5_PolynomialRegression.html

# Load required libraries
library(ISLR)
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Load the Wage dataset
data(Wage)

# Extract age and wage variables
age <- Wage$age
wage <- Wage$wage

# Create polynomial regression models
# Model 1: Direct polynomial terms
fit1 <- lm(wage ~ age + I(age^2) + I(age^3), data = Wage)

# Model 2: Using poly() function for orthogonal polynomials
fit2 <- lm(wage ~ poly(age, 3), data = Wage)

# Display model summaries
cat("Model 1 (Direct polynomial terms):\n")
print(summary(fit1))

cat("\nModel 2 (Orthogonal polynomials):\n")
print(summary(fit2))

# Compare coefficients
cat("\nCoefficients comparison:\n")
cat("Model 1 coefficients:\n")
print(coef(fit1))

cat("\nModel 2 coefficients:\n")
print(coef(fit2))

# Note: While coefficients are different, the t-values and p-values for the last predictor remain identical
# This is because in linear regression, the t-test for a variable gauges its conditional contribution
# In both models, the sole unique contribution of the cubic term of age comes only from the last variable
# Thus, the p-value for the last variable in both models stays the same

# Prediction
# Predict the wage at age = 82
cat("\nPrediction at age = 82:\n")
prediction_result <- c(predict(fit1, newdata = list(age = 82)), 
                      predict(fit2, newdata = list(age = 82)))
print(prediction_result)

# Both models should give the same answer for predictions

# Create fitted curve plot
agelims <- range(age)
age.grid <- seq(from = agelims[1], to = agelims[2])

# Get predictions with standard errors
# preds = predict(fit1, newdata = list(age = age.grid), se=TRUE)
preds <- predict(fit2, newdata = list(age = age.grid), se = TRUE)

# Create the plot
plot(age, wage, xlim = agelims, pch = '.', cex = 2, col = "darkgrey")
title("Degree-3 Polynomial")
lines(age.grid, preds$fit, lwd = 2, col = "blue")

# Add confidence intervals
lines(age.grid, preds$fit + 2 * preds$se.fit, lwd = 1, col = "blue", lty = 2)
lines(age.grid, preds$fit - 2 * preds$se.fit, lwd = 1, col = "blue", lty = 2)

# Alternative plotting using ggplot2
library(ggplot2)

# Create data frame for plotting
plot_data <- data.frame(
  age = age.grid,
  fit = preds$fit,
  upper = preds$fit + 2 * preds$se.fit,
  lower = preds$fit - 2 * preds$se.fit
)

# Create ggplot
ggplot() +
  geom_point(data = data.frame(age = age, wage = wage), 
             aes(x = age, y = wage), alpha = 0.3, size = 1) +
  geom_line(data = plot_data, aes(x = age, y = fit), 
            color = "blue", size = 1) +
  geom_ribbon(data = plot_data, 
              aes(x = age, ymin = lower, ymax = upper), 
              alpha = 0.2, fill = "blue") +
  labs(title = "Degree-3 Polynomial Regression",
       x = "Age", y = "Wage") +
  theme_minimal()

# Model comparison using ANOVA
cat("\nModel comparison using ANOVA:\n")
anova_result <- anova(fit1, fit2)
print(anova_result)

# Residual analysis
cat("\nResidual analysis for Model 2:\n")
residuals_fit2 <- residuals(fit2)
fitted_fit2 <- fitted(fit2)

# Plot residuals vs fitted values
plot(fitted_fit2, residuals_fit2, 
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red", lty = 2)

# Q-Q plot for normality
qqnorm(residuals_fit2)
qqline(residuals_fit2, col = "red")

# Histogram of residuals
hist(residuals_fit2, main = "Histogram of Residuals", 
     xlab = "Residuals", probability = TRUE)
lines(density(residuals_fit2), col = "red")

# Summary statistics of residuals
cat("\nResidual summary statistics:\n")
print(summary(residuals_fit2))
cat("Standard deviation of residuals:", sd(residuals_fit2), "\n")

# R-squared and adjusted R-squared
cat("\nModel fit statistics:\n")
cat("R-squared:", summary(fit2)$r.squared, "\n")
cat("Adjusted R-squared:", summary(fit2)$adj.r.squared, "\n")

# Cross-validation for model selection
library(boot)

# Function to calculate MSE for cross-validation
cv_mse <- function(data, formula, degree) {
  cv.error <- rep(0, degree)
  for (i in 1:degree) {
    glm.fit <- glm(formula, data = data)
    cv.error[i] <- cv.glm(data, glm.fit)$delta[1]
  }
  return(cv.error)
}

# Perform cross-validation for different polynomial degrees
cat("\nCross-validation for polynomial degree selection:\n")
cv_errors <- cv_mse(Wage, wage ~ poly(age, 3), 3)
for (i in 1:3) {
  cat("Degree", i, "CV MSE:", cv_errors[i], "\n")
}

# Find optimal degree
optimal_degree <- which.min(cv_errors)
cat("Optimal polynomial degree:", optimal_degree, "\n")

# Fit optimal model
optimal_fit <- lm(wage ~ poly(age, optimal_degree), data = Wage)
cat("\nOptimal model summary:\n")
print(summary(optimal_fit))

# Final prediction with optimal model
cat("\nFinal prediction at age = 82 (optimal model):\n")
final_prediction <- predict(optimal_fit, newdata = list(age = 82))
print(final_prediction)

# Save the models for later use
save(fit1, fit2, optimal_fit, file = "polynomial_models.RData")

cat("\nAnalysis complete. Models saved to 'polynomial_models.RData'\n") 