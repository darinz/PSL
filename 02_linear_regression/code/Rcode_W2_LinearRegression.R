# =============================================================================
# Advanced Linear Regression Analysis in R
# =============================================================================
# 
# This script demonstrates comprehensive linear regression analysis using R,
# covering fundamental concepts to advanced topics in statistical learning.
#
# Key Learning Objectives:
# - Data loading, exploration, and preprocessing
# - Linear model fitting and interpretation
# - Prediction methods and validation
# - Understanding rank deficiency and multicollinearity
# - Training vs test error analysis
# - Coefficient interpretation and partial effects
# - Hypothesis testing and model comparison
# - Collinearity detection and handling
#
# =============================================================================

# =============================================================================
# Load Required Libraries
# =============================================================================

# Load essential packages for data analysis and visualization
library(faraway)  # For additional datasets
library(car)      # For additional diagnostic tools

# Set plotting parameters for better visualizations
par(mar = c(4, 4, 2, 1))  # Adjust margins for plots
options(digits = 4)        # Set number of digits for output

# =============================================================================
# Data Loading and Exploration
# =============================================================================

cat("=" * 60, "\n")
cat("LOADING PROSTATE CANCER DATASET\n")
cat("=" * 60, "\n")

# Load the Prostate cancer dataset from ESL website
# This dataset examines the correlation between prostate-specific antigen (PSA)
# levels and various clinical measures in men about to receive radical prostatectomy
prostate_url <- "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
prostate_data <- read.table(file = prostate_url, header = TRUE)

# Display basic information about the dataset
cat("Dataset dimensions:", dim(prostate_data), "\n")
cat("Variable names:", names(prostate_data), "\n\n")

# Variable descriptions for educational purposes
cat("Variable Descriptions:\n")
cat("- lcavol:  log cancer volume\n")
cat("- lweight: log prostate weight\n")
cat("- age:     age in years\n")
cat("- lbph:    log of benign prostatic hyperplasia\n")
cat("- svi:     seminal vesicle invasion (binary)\n")
cat("- lcp:     log of capsular penetration\n")
cat("- gleason: Gleason score\n")
cat("- pgg45:   percent of Gleason score 4 or 5\n")
cat("- lpsa:    log PSA (response variable)\n")
cat("- train:   training/test indicator (will be removed)\n\n")

# Data preprocessing: Remove the training indicator column and rename response
prostate_data <- prostate_data[, -10]  # Remove column 10 (train/test indicator)
names(prostate_data)[9] <- 'Y'         # Rename response to generic 'Y'

cat("After preprocessing:\n")
cat("Dimensions:", dim(prostate_data), "\n")
cat("Variables:", names(prostate_data), "\n\n")

# Comprehensive data summary
cat("Summary Statistics:\n")
print(summary(prostate_data))

# =============================================================================
# Exploratory Data Analysis
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("EXPLORATORY DATA ANALYSIS\n")
cat("=" * 60, "\n")

# Create pairwise scatter plots to visualize relationships
cat("Creating pairwise scatter plots...\n")
cat("Note: This creates a large figure showing all variable relationships\n\n")

# Create a more informative pairs plot with correlation coefficients
pairs(prostate_data, 
      pch = 16,           # Solid circles for better visibility
      col = "steelblue",  # Consistent color scheme
      cex = 0.8,          # Slightly smaller points
      main = "Pairwise Scatter Plots - Prostate Cancer Data")

# Calculate and display correlation matrix
cat("Correlation Matrix:\n")
correlation_matrix <- round(cor(prostate_data), digits = 3)
print(correlation_matrix)

# Find high correlations for educational purposes
cat("\nHigh correlations (|r| > 0.7):\n")
for(i in 1:(ncol(correlation_matrix)-1)) {
  for(j in (i+1):ncol(correlation_matrix)) {
    cor_val <- correlation_matrix[i, j]
    if(abs(cor_val) > 0.7) {
      cat(sprintf("%s - %s: %.3f\n", 
                  names(prostate_data)[i], 
                  names(prostate_data)[j], 
                  cor_val))
    }
  }
}

# =============================================================================
# Linear Model Fitting
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("LINEAR MODEL FITTING\n")
cat("=" * 60, "\n")

# Fit a linear regression model using all predictors
cat("Fitting linear regression model with all predictors...\n")
lm_fit <- lm(Y ~ ., data = prostate_data)

# Display comprehensive model summary
cat("\nModel Summary:\n")
model_summary <- summary(lm_fit)
print(model_summary)

# Extract key model components for educational purposes
cat("\nModel Components:\n")
cat("Available components from lm() object:\n")
print(names(lm_fit))

cat("\nAvailable components from summary():\n")
print(names(model_summary))

# Extract and display key statistics
cat("\nKey Model Statistics:\n")
cat("Number of observations:", length(lm_fit$residuals), "\n")
cat("Number of predictors:", length(lm_fit$coefficients) - 1, "\n")
cat("Degrees of freedom:", lm_fit$df.residual, "\n")

# =============================================================================
# Manual Calculation Verification
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("MANUAL CALCULATION VERIFICATION\n")
cat("=" * 60, "\n")

# Educational purpose: Verify R's calculations manually
cat("Verifying R's calculations manually for educational purposes:\n\n")

# Get dimensions
n <- nrow(prostate_data)  # Number of observations
p <- ncol(prostate_data) - 1  # Number of predictors (excluding response)

cat("Sample size (n):", n, "\n")
cat("Number of predictors (p):", p, "\n\n")

# Manual calculation of residual standard error
# Formula: σ = √(Σr²/(n-p-1))
manual_sigma <- sqrt(sum(lm_fit$residuals^2) / (n - p - 1))
r_sigma <- model_summary$sigma

cat("Residual Standard Error:\n")
cat("Manual calculation:", round(manual_sigma, 6), "\n")
cat("R's calculation:   ", round(r_sigma, 6), "\n")
cat("Difference:        ", round(abs(manual_sigma - r_sigma), 10), "\n\n")

# Manual calculation of R-squared
# Formula: R² = 1 - (Σr²)/(Σ(y-ȳ)²)
manual_r_squared <- 1 - sum(lm_fit$residuals^2) / (var(prostate_data$Y) * (n - 1))
r_r_squared <- model_summary$r.squared

cat("R-squared:\n")
cat("Manual calculation:", round(manual_r_squared, 6), "\n")
cat("R's calculation:   ", round(r_r_squared, 6), "\n")
cat("Difference:        ", round(abs(manual_r_squared - r_r_squared), 10), "\n\n")

# Alternative R-squared calculation using variance ratio
alt_r_squared <- 1 - var(lm_fit$residuals) / var(prostate_data$Y)
cat("Alternative R-squared calculation:", round(alt_r_squared, 6), "\n")

# =============================================================================
# Prediction Methods
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("PREDICTION METHODS\n")
cat("=" * 60, "\n")

cat("Demonstrating two methods for making predictions:\n\n")

# Method 1: Manual calculation using coefficients
cat("Method 1: Manual calculation using coefficients\n")
cat("Formula: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ\n\n")

# Create new observations for prediction
# Patient 1: 50 years old, other variables at median values
# Patient 2: 70 years old, other variables at median values
median_values <- apply(prostate_data, 2, median)
cat("Median values for each variable:\n")
print(median_values)

# Create prediction data
new_patient_50 <- c(1.45, 3.62, 50, 0.30, 0, -0.80, 7, 15)  # Age = 50
new_patient_70 <- new_patient_50
new_patient_70[3] <- 70  # Age = 70

# Manual prediction using matrix multiplication
new_data_matrix <- rbind(new_patient_50, new_patient_70)
manual_predictions <- new_data_matrix %*% lm_fit$coefficients[-1] + lm_fit$coefficients[1]

cat("\nManual predictions:\n")
cat("Patient 1 (age 50):", round(manual_predictions[1], 4), "\n")
cat("Patient 2 (age 70):", round(manual_predictions[2], 4), "\n\n")

# Method 2: Using R's predict() function
cat("Method 2: Using R's predict() function\n")
cat("Advantage: Column order doesn't matter when names are provided\n\n")

# Create data frame for prediction
new_data_df <- data.frame(new_data_matrix)
names(new_data_df) <- names(prostate_data[, -ncol(prostate_data)])

cat("Original column order:\n")
print(new_data_df)

# Shuffle columns to demonstrate order independence
set.seed(123)  # For reproducibility
shuffled_cols <- sample(1:ncol(new_data_df))
new_data_shuffled <- new_data_df[, shuffled_cols]

cat("\nShuffled column order:\n")
print(new_data_shuffled)

# Make predictions using predict() function
r_predictions <- predict(lm_fit, newdata = new_data_shuffled)

cat("\nR predict() function results:\n")
cat("Patient 1 (age 50):", round(r_predictions[1], 4), "\n")
cat("Patient 2 (age 70):", round(r_predictions[2], 4), "\n\n")

# Verify both methods give same results
cat("Verification - Both methods should give identical results:\n")
cat("Method 1 vs Method 2 differences:\n")
cat("Patient 1:", round(abs(manual_predictions[1] - r_predictions[1]), 10), "\n")
cat("Patient 2:", round(abs(manual_predictions[2] - r_predictions[2]), 10), "\n")

# =============================================================================
# Rank Deficiency Analysis
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("RANK DEFICIENCY ANALYSIS\n")
cat("=" * 60, "\n")

cat("Demonstrating what happens when the design matrix is not full rank:\n\n")

# Create a redundant variable (linear combination of existing variables)
cat("Adding a redundant variable 'junk' = lcavol + lweight\n")
prostate_data$junk <- prostate_data[, 1] + prostate_data[, 2]

# Fit model with redundant variable
cat("Fitting model with redundant variable...\n")
rank_deficient_fit <- lm(Y ~ ., data = prostate_data)
rank_summary <- summary(rank_deficient_fit)

cat("\nModel summary with redundant variable:\n")
print(rank_summary)

# Check for NA coefficients
na_coefficients <- is.na(rank_deficient_fit$coefficients)
if(any(na_coefficients)) {
  cat("\nNA coefficients found for variables:\n")
  print(names(rank_deficient_fit$coefficients)[na_coefficients])
  cat("\nThis indicates rank deficiency - these variables are redundant.\n")
} else {
  cat("\nNo NA coefficients found.\n")
}

# Compare fitted values from both models
cat("\nComparing fitted values from both models:\n")
cat("Original model (first 3 observations):\n")
print(lm_fit$fitted[1:3])

cat("\nModel with redundant variable (first 3 observations):\n")
print(rank_deficient_fit$fitted[1:3])

cat("\nDifferences (should be zero):\n")
print(lm_fit$fitted[1:3] - rank_deficient_fit$fitted[1:3])

# Remove the redundant variable
cat("\nRemoving redundant variable...\n")
prostate_data <- prostate_data[, !names(prostate_data) == 'junk']

cat("Key insights:\n")
cat("- NA coefficients indicate rank deficiency\n")
cat("- Redundant variables don't affect predictions\n")
cat("- R automatically handles rank deficiency\n")
cat("- Model utility is preserved despite NA coefficients\n")

# =============================================================================
# Training vs Test Error Analysis
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("TRAINING VS TEST ERROR ANALYSIS\n")
cat("=" * 60, "\n")

cat("Demonstrating the bias-variance tradeoff:\n")
cat("- Training error decreases with more predictors\n")
cat("- Test error may increase, indicating overfitting\n\n")

# Set up parameters
n <- nrow(prostate_data)
p <- ncol(prostate_data) - 1
ntrain <- round(n * 0.6)  # 60% training, 40% test

cat("Sample size:", n, "\n")
cat("Number of predictors:", p, "\n")
cat("Training set size:", ntrain, "\n")
cat("Test set size:", n - ntrain, "\n\n")

# Set seed for reproducibility
set.seed(42)

# Split data into training and test sets
train_indices <- sample(1:n, ntrain)
train_data <- prostate_data[train_indices, ]
test_data <- prostate_data[-train_indices, ]

# Arrays to store errors
train_MSE <- numeric(p)
test_MSE <- numeric(p)

# Fit models with progressively more predictors
cat("Fitting models with increasing numbers of predictors...\n")
for(i in 1:p) {
  # Create formula with i predictors
  predictor_names <- names(prostate_data)[1:i]
  formula_str <- paste("Y ~", paste(predictor_names, collapse = " + "))
  
  # Fit model on training data
  current_fit <- lm(as.formula(formula_str), data = train_data)
  
  # Calculate training MSE
  train_predictions <- current_fit$fitted
  train_actual <- train_data$Y
  train_MSE[i] <- mean((train_actual - train_predictions)^2)
  
  # Calculate test MSE
  test_predictions <- predict(current_fit, newdata = test_data)
  test_actual <- test_data$Y
  test_MSE[i] <- mean((test_actual - test_predictions)^2)
  
  cat(sprintf("Predictors %d: Train MSE = %.4f, Test MSE = %.4f\n", 
              i, train_MSE[i], test_MSE[i]))
}

# Create visualization
cat("\nCreating training vs test error plot...\n")
plot(1:p, train_MSE, 
     type = "b", 
     col = "blue", 
     pch = 16, 
     lwd = 2,
     xlab = "Number of Predictors", 
     ylab = "Mean Squared Error",
     main = "Training vs Test Error",
     ylim = range(train_MSE, test_MSE))

lines(1:p, test_MSE, 
      type = "b", 
      col = "red", 
      pch = 17, 
      lwd = 2)

legend("topright", 
       legend = c("Training Error", "Test Error"),
       col = c("blue", "red"), 
       pch = c(16, 17), 
       lwd = 2)

grid()

# Analyze the differences
cat("\nAnalysis of error patterns:\n")
train_diff <- diff(train_MSE)
test_diff <- diff(test_MSE)

cat("Training error changes (should all be negative):\n")
print(round(train_diff, 6))
cat("All training differences negative:", all(train_diff <= 0), "\n\n")

cat("Test error changes (may be positive):\n")
print(round(test_diff, 6))
cat("Any test differences positive:", any(test_diff > 0), "\n")
cat("Number of test error increases:", sum(test_diff > 0), "\n\n")

cat("Key insights:\n")
cat("- Training error always decreases (monotonic)\n")
cat("- Test error may increase, indicating overfitting\n")
cat("- Optimal model complexity balances bias and variance\n")
cat("- Cross-validation helps identify optimal complexity\n")

# =============================================================================
# Coefficient Interpretation Analysis
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("COEFFICIENT INTERPRETATION ANALYSIS\n")
cat("=" * 60, "\n")

cat("Comparing Simple vs Multiple Linear Regression:\n\n")

# Simple Linear Regression with age only
cat("1. Simple Linear Regression (age only):\n")
simple_age_fit <- lm(Y ~ age, data = prostate_data)
simple_summary <- summary(simple_age_fit)

cat("Age coefficient:", round(simple_age_fit$coefficients["age"], 6), "\n")
cat("P-value:", round(simple_summary$coefficients["age", "Pr(>|t|)"], 6), "\n")
cat("R-squared:", round(simple_summary$r.squared, 6), "\n\n")

# Multiple Linear Regression with all predictors
cat("2. Multiple Linear Regression (all predictors):\n")
multiple_summary <- summary(lm_fit)

age_coef_multiple <- lm_fit$coefficients["age"]
age_p_multiple <- multiple_summary$coefficients["age", "Pr(>|t|)"]

cat("Age coefficient:", round(age_coef_multiple, 6), "\n")
cat("P-value:", round(age_p_multiple, 6), "\n")
cat("R-squared:", round(multiple_summary$r.squared, 6), "\n\n")

# Compare coefficients
cat("3. Coefficient Comparison:\n")
cat("Simple regression age coefficient:  ", round(simple_age_fit$coefficients["age"], 6), "\n")
cat("Multiple regression age coefficient: ", round(age_coef_multiple, 6), "\n")
cat("Difference:                        ", round(age_coef_multiple - simple_age_fit$coefficients["age"], 6), "\n\n")

# Correlation analysis
cat("4. Correlation Analysis:\n")
age_correlations <- cor(prostate_data)[, "age"]
age_correlations <- age_correlations[age_correlations != 1]  # Remove self-correlation
age_correlations <- sort(age_correlations, decreasing = TRUE)

cat("Correlations with age:\n")
for(i in 1:length(age_correlations)) {
  cat(sprintf("  %-10s: %8.3f\n", names(age_correlations)[i], age_correlations[i]))
}

cat("\n5. Interpretation:\n")
cat("- Simple regression captures total effect of age\n")
cat("- Multiple regression captures partial effect of age\n")
cat("- Difference due to correlations with other variables\n")
cat("- Multiple regression coefficients are 'partial effects'\n")

# =============================================================================
# Partial Regression Coefficient (Frisch-Waugh-Lovell)
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("PARTIAL REGRESSION COEFFICIENT ANALYSIS\n")
cat("=" * 60, "\n")

cat("Implementing the Frisch-Waugh-Lovell theorem:\n")
cat("This demonstrates how to isolate the effect of one predictor\n\n")

# Step 1: Regress Y on all predictors except age
cat("Step 1: Regress Y on all predictors except age\n")
y_residuals <- lm(Y ~ ., data = prostate_data[, -which(names(prostate_data) == "age")])$residuals

cat("Y residuals (y*):\n")
cat("Mean:", round(mean(y_residuals), 6), "\n")
cat("Std:", round(sd(y_residuals), 6), "\n")
cat("Range:", round(range(y_residuals), 6), "\n\n")

# Step 2: Regress age on all other predictors
cat("Step 2: Regress age on all other predictors\n")
age_residuals <- lm(age ~ ., data = prostate_data[, -which(names(prostate_data) == "Y")])$residuals

cat("Age residuals (age*):\n")
cat("Mean:", round(mean(age_residuals), 6), "\n")
cat("Std:", round(sd(age_residuals), 6), "\n")
cat("Range:", round(range(age_residuals), 6), "\n\n")

# Step 3: Regress y* on age*
cat("Step 3: Regress y* on age*\n")
partial_fit <- lm(y_residuals ~ age_residuals)
partial_coefficient <- partial_fit$coefficients["age_residuals"]

cat("Partial regression coefficient:", round(partial_coefficient, 8), "\n\n")

# Compare with full model
cat("Comparison with full model:\n")
full_coefficient <- lm_fit$coefficients["age"]
cat("Partial regression coefficient:", round(partial_coefficient, 8), "\n")
cat("Full model coefficient:        ", round(full_coefficient, 8), "\n")
cat("Difference:                    ", round(abs(partial_coefficient - full_coefficient), 10), "\n")
cat("Are they equal?                ", isTRUE(all.equal(partial_coefficient, full_coefficient)), "\n\n")

# Verify residuals are the same
cat("Residual comparison:\n")
full_residuals <- lm_fit$residuals
partial_residuals <- partial_fit$residuals

cat("Full model residuals sum of squares:", round(sum(full_residuals^2), 8), "\n")
cat("Partial model residuals sum of squares:", round(sum(partial_residuals^2), 8), "\n")
cat("Difference:", round(abs(sum(full_residuals^2) - sum(partial_residuals^2)), 10), "\n\n")

cat("Interpretation:\n")
cat("- The partial coefficient isolates the effect of age\n")
cat("- It removes the influence of other variables\n")
cat("- This is why multiple regression coefficients are 'partial effects'\n")
cat("- The theorem shows that partial and full model coefficients are identical\n")

# =============================================================================
# Hypothesis Testing (F-test and t-test)
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("HYPOTHESIS TESTING ANALYSIS\n")
cat("=" * 60, "\n")

cat("Testing the significance of individual predictors:\n\n")

# Test single predictor (age)
cat("1. Testing single predictor (age):\n")
cat("H₀: β_age = 0 vs H₁: β_age ≠ 0\n\n")

# Fit reduced model (without age)
reduced_fit <- lm(Y ~ ., data = prostate_data[, -which(names(prostate_data) == "age")])

# Perform F-test using anova()
f_test_result <- anova(reduced_fit, lm_fit)
cat("F-test results:\n")
print(f_test_result)

# Extract key statistics
f_statistic <- f_test_result$F[2]
f_p_value <- f_test_result$`Pr(>F)`[2]

cat("\nF-statistic:", round(f_statistic, 6), "\n")
cat("P-value:", round(f_p_value, 6), "\n")
cat("Degrees of freedom:", f_test_result$Df[2], ",", f_test_result$Res.Df[2], "\n\n")

# Compare with t-test (for single variable, F = t²)
t_statistic <- sqrt(f_statistic)
t_p_value <- 2 * (1 - pt(abs(t_statistic), df = lm_fit$df.residual))

cat("t-test comparison:\n")
cat("t-statistic:", round(t_statistic, 6), "\n")
cat("P-value:", round(t_p_value, 6), "\n")
cat("F = t² verification:", round(f_statistic, 6), "=", round(t_statistic^2, 6), "\n\n")

# Test multiple predictors
cat("2. Testing multiple predictors (first 3 variables):\n")
cat("H₀: β₁ = β₂ = β₃ = 0 vs H₁: At least one βᵢ ≠ 0\n\n")

# Fit reduced model (without first 3 variables)
reduced_fit_multi <- lm(Y ~ ., data = prostate_data[, -c(1:3)])

# Perform F-test
f_test_multi <- anova(reduced_fit_multi, lm_fit)
cat("F-test results for multiple variables:\n")
print(f_test_multi)

cat("\nKey insights:\n")
cat("- F-test and t-test give identical results for single variables\n")
cat("- F = t² for single variable tests\n")
cat("- Multiple variable tests use F-distribution with multiple df\n")
cat("- F-test is more general and can test multiple hypotheses\n")

# =============================================================================
# Collinearity Analysis
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("COLLINEARITY ANALYSIS\n")
cat("=" * 60, "\n")

cat("Analyzing the Car Seat Position dataset to demonstrate multicollinearity:\n\n")

# Load seat position data
cat("Loading seat position data...\n")
data(seatpos)

cat("Dataset dimensions:", dim(seatpos), "\n")
cat("Variables:", names(seatpos), "\n\n")

# Variable descriptions
cat("Variable descriptions:\n")
cat("- Age: Driver age in years\n")
cat("- Weight: Driver weight in kg\n")
cat("- HtShoes: Height with shoes in cm\n")
cat("- Ht: Height without shoes in cm\n")
cat("- Seated: Seated height in cm\n")
cat("- Arm: Lower arm length in cm\n")
cat("- Thigh: Thigh length in cm\n")
cat("- Leg: Lower leg length in cm\n")
cat("- hipcenter: Horizontal distance of hip center (mm)\n\n")

# Data exploration
cat("Summary statistics:\n")
print(summary(seatpos))

# Correlation analysis
cat("\nCorrelation matrix:\n")
seatpos_cor <- round(cor(seatpos), digits = 2)
print(seatpos_cor)

# Find high correlations
cat("\nHigh correlations (|r| > 0.8):\n")
high_corr_pairs <- list()
for(i in 1:(ncol(seatpos_cor)-1)) {
  for(j in (i+1):ncol(seatpos_cor)) {
    cor_val <- seatpos_cor[i, j]
    if(abs(cor_val) > 0.8) {
      high_corr_pairs[[length(high_corr_pairs) + 1]] <- 
        list(var1 = names(seatpos)[i], var2 = names(seatpos)[j], corr = cor_val)
    }
  }
}

for(pair in high_corr_pairs) {
  cat(sprintf("  %s - %s: %.2f\n", pair$var1, pair$var2, pair$corr))
}

# Fit full model
cat("\nFitting full model with all predictors:\n")
full_seatpos_fit <- lm(hipcenter ~ ., data = seatpos)
full_seatpos_summary <- summary(full_seatpos_fit)

cat("Full model results:\n")
cat("R-squared:", round(full_seatpos_summary$r.squared, 6), "\n")
cat("Adjusted R-squared:", round(full_seatpos_summary$adj.r.squared, 6), "\n")
cat("F-statistic:", round(full_seatpos_summary$fstatistic[1], 6), "\n")
cat("F-test p-value:", round(pf(full_seatpos_summary$fstatistic[1], 
                                full_seatpos_summary$fstatistic[2], 
                                full_seatpos_summary$fstatistic[3], 
                                lower.tail = FALSE), 6), "\n\n")

# Check individual coefficient significance
cat("Individual coefficient significance:\n")
coef_table <- full_seatpos_summary$coefficients
significant_vars <- rownames(coef_table)[coef_table[, "Pr(>|t|)"] < 0.05]

cat("Significant variables (p < 0.05):", if(length(significant_vars) > 0) significant_vars else "None", "\n")
cat("Number of significant variables:", length(significant_vars), "out of", ncol(seatpos) - 1, "\n\n")

# Try reduced models
cat("Trying reduced models to improve interpretability:\n\n")

# Model with only height
height_fit <- lm(hipcenter ~ Ht, data = seatpos)
height_summary <- summary(height_fit)

# Model with height and weight
height_weight_fit <- lm(hipcenter ~ Ht + Weight, data = seatpos)
height_weight_summary <- summary(height_weight_fit)

# Model with height, weight, and age
height_weight_age_fit <- lm(hipcenter ~ Ht + Weight + Age, data = seatpos)
height_weight_age_summary <- summary(height_weight_age_fit)

# Compare models
cat("Model comparison:\n")
cat(sprintf("%-25s %-10s %-10s %-10s %-10s\n", "Model", "R²", "Adj R²", "F-stat", "P-value"))
cat(paste(rep("-", 65), collapse = ""), "\n")

models <- list(
  "Height only" = height_fit,
  "Height + Weight" = height_weight_fit,
  "Height + Weight + Age" = height_weight_age_fit,
  "All variables" = full_seatpos_fit
)

for(name in names(models)) {
  model <- models[[name]]
  summary_model <- summary(model)
  f_stat <- summary_model$fstatistic[1]
  f_p_val <- pf(f_stat, summary_model$fstatistic[2], summary_model$fstatistic[3], lower.tail = FALSE)
  
  cat(sprintf("%-25s %-10.4f %-10.4f %-10.4f %-10.4f\n",
              name, summary_model$r.squared, summary_model$adj.r.squared, f_stat, f_p_val))
}

cat("\nKey insights:\n")
cat("- High R² but few significant coefficients indicates multicollinearity\n")
cat("- Removing correlated variables often improves interpretability\n")
cat("- Simple models may be more useful than complex ones\n")
cat("- Variable selection is important in the presence of multicollinearity\n")

# =============================================================================
# Summary and Conclusions
# =============================================================================

cat("\n" + "=" * 70, "\n")
cat("ANALYSIS SUMMARY AND CONCLUSIONS\n")
cat("=" * 70, "\n")

cat("Key Concepts Demonstrated:\n\n")

cat("1. Data Loading and Exploration\n")
cat("   • Proper data preprocessing and cleaning\n")
cat("   • Exploratory data analysis with visualizations\n")
cat("   • Understanding variable relationships\n\n")

cat("2. Linear Model Fitting\n")
cat("   • Model specification and interpretation\n")
cat("   • Manual verification of statistical calculations\n")
cat("   • Understanding model components and outputs\n\n")

cat("3. Prediction Methods\n")
cat("   • Manual prediction using coefficients\n")
cat("   • Using R's predict() function\n")
cat("   • Handling new data with proper formatting\n\n")

cat("4. Rank Deficiency\n")
cat("   • Understanding when design matrix is not full rank\n")
cat("   • How R handles redundant variables\n")
cat("   • Impact on model interpretation\n\n")

cat("5. Bias-Variance Tradeoff\n")
cat("   • Training vs test error analysis\n")
cat("   • Overfitting demonstration\n")
cat("   • Model complexity selection\n\n")

cat("6. Coefficient Interpretation\n")
cat("   • Simple vs multiple linear regression\n")
cat("   • Partial effects vs total effects\n")
cat("   • Impact of multicollinearity\n\n")

cat("7. Partial Regression Coefficients\n")
cat("   • Frisch-Waugh-Lovell theorem implementation\n")
cat("   • Mathematical foundation of partial effects\n")
cat("   • Isolating individual predictor effects\n\n")

cat("8. Hypothesis Testing\n")
cat("   • F-test and t-test for individual predictors\n")
cat("   • Testing multiple variables simultaneously\n")
cat("   • Understanding test statistics and p-values\n\n")

cat("9. Multicollinearity\n")
cat("   • Detection methods and interpretation\n")
cat("   • Effects on coefficient significance\n")
cat("   • Strategies for handling collinearity\n\n")

cat("Practical Applications:\n")
cat("• Use cross-validation to find optimal model complexity\n")
cat("• Interpret coefficients as partial effects\n")
cat("• Check for multicollinearity before interpreting coefficients\n")
cat("• Consider variable selection or regularization techniques\n")
cat("• Understand the difference between statistical and practical significance\n")

cat("\n" + "=" * 70, "\n")
cat("END OF LINEAR REGRESSION ANALYSIS\n")
cat("=" * 70, "\n") 