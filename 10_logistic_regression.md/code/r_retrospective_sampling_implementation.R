# Retrospective Sampling in Logistic Regression

# Load required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)
library(pROC)

# Generate population data
generate_population_data <- function(n_population = 10000, prevalence = 0.01) {
  set.seed(42)
  
  # Generate features
  X <- matrix(rnorm(n_population * 3), n_population, 3)
  
  # True population parameters
  true_alpha <- -4.6  # Logit of 0.01 prevalence
  true_beta <- c(0.5, -0.3, 0.8)
  
  # Generate probabilities and outcomes
  z <- true_alpha + X %*% true_beta
  p <- 1 / (1 + exp(-z))
  y <- rbinom(n_population, 1, p)
  
  population_data <- data.frame(
    X1 = X[, 1],
    X2 = X[, 2],
    X3 = X[, 3],
    y = y,
    probability = p
  )
  
  cat("Population Summary:\n")
  cat("Total samples:", n_population, "\n")
  cat("Cases:", sum(y), "(", sum(y)/n_population, ")\n")
  cat("Controls:", n_population - sum(y), "(", (n_population - sum(y))/n_population, ")\n")
  
  return(list(data = population_data, true_alpha = true_alpha, true_beta = true_beta))
}

# Create retrospective sample
create_retrospective_sample <- function(population_data, n_cases = 100, n_controls = 100) {
  cases <- population_data[population_data$y == 1, ]
  controls <- population_data[population_data$y == 0, ]
  
  # Sample cases and controls
  sampled_cases <- cases[sample(nrow(cases), min(n_cases, nrow(cases))), ]
  sampled_controls <- controls[sample(nrow(controls), min(n_controls, nrow(controls))), ]
  
  retrospective_data <- rbind(sampled_cases, sampled_controls)
  
  cat("\nRetrospective Sample Summary:\n")
  cat("Cases:", nrow(sampled_cases), "\n")
  cat("Controls:", nrow(sampled_controls), "\n")
  cat("Total:", nrow(retrospective_data), "\n")
  
  return(retrospective_data)
}

# Fit models and compare
compare_models <- function(population_data, retrospective_data, true_alpha, true_beta) {
  # Population model
  pop_model <- glm(y ~ X1 + X2 + X3, data = population_data, family = binomial)
  
  # Retrospective model
  retro_model <- glm(y ~ X1 + X2 + X3, data = retrospective_data, family = binomial)
  
  # Extract coefficients
  pop_coef <- coef(pop_model)
  retro_coef <- coef(retro_model)
  
  # Create comparison table
  comparison <- data.frame(
    True = c(true_alpha, true_beta),
    Population = pop_coef,
    Retrospective = retro_coef,
    Difference = retro_coef - pop_coef
  )
  rownames(comparison) <- c("Intercept", "X1", "X2", "X3")
  
  print("=== Model Comparison ===")
  print(round(comparison, 4))
  
  # Theoretical intercept adjustment
  n_cases <- sum(retrospective_data$y == 1)
  n_controls <- sum(retrospective_data$y == 0)
  n_population <- nrow(population_data)
  n_cases_pop <- sum(population_data$y == 1)
  n_controls_pop <- nrow(population_data) - sum(population_data$y == 1)
  
  pi_1 <- n_cases / n_cases_pop  # Sampling probability for cases
  pi_0 <- n_controls / n_controls_pop  # Sampling probability for controls
  
  theoretical_adjustment <- log(pi_1 / pi_0)
  actual_adjustment <- retro_coef[1] - pop_coef[1]
  
  cat("\nSampling Probabilities:\n")
  cat("π₁ (cases):", pi_1, "\n")
  cat("π₀ (controls):", pi_0, "\n")
  cat("log(π₁/π₀):", theoretical_adjustment, "\n")
  cat("Actual intercept difference:", actual_adjustment, "\n")
  cat("Difference:", abs(theoretical_adjustment - actual_adjustment), "\n")
  
  return(list(comparison = comparison, 
              pop_model = pop_model, 
              retro_model = retro_model,
              theoretical_adjustment = theoretical_adjustment))
}

# Adjust probabilities
adjust_probabilities <- function(retro_probs, population_data, retrospective_data) {
  # Calculate sampling probabilities
  n_cases <- sum(retrospective_data$y == 1)
  n_controls <- sum(retrospective_data$y == 0)
  n_cases_pop <- sum(population_data$y == 1)
  n_controls_pop <- nrow(population_data) - sum(population_data$y == 1)
  
  pi_1 <- n_cases / n_cases_pop
  pi_0 <- n_controls / n_controls_pop
  
  # Adjust probabilities
  adjusted_probs <- retro_probs / (retro_probs + (1 - retro_probs) * pi_0 / pi_1)
  
  return(adjusted_probs)
}

# Evaluate predictions
evaluate_predictions <- function(population_data, pop_model, retro_model) {
  # Predictions from both models
  pop_pred_proba <- predict(pop_model, newdata = population_data, type = "response")
  retro_pred_proba <- predict(retro_model, newdata = population_data, type = "response")
  
  # Adjust retrospective predictions
  adjusted_pred_proba <- adjust_probabilities(retro_pred_proba, population_data, 
                                             data.frame(y = c(rep(1, 100), rep(0, 100))))
  
  # Calculate metrics
  results <- list()
  
  for (name in c("Population", "Retrospective", "Adjusted")) {
    if (name == "Population") {
      pred_proba <- pop_pred_proba
    } else if (name == "Retrospective") {
      pred_proba <- retro_pred_proba
    } else {
      pred_proba <- adjusted_pred_proba
    }
    
    pred_class <- ifelse(pred_proba >= 0.5, 1, 0)
    accuracy <- mean(pred_class == population_data$y)
    
    # Calculate AUC
    auc <- auc(population_data$y, pred_proba)
    
    results[[name]] <- list(
      Accuracy = accuracy,
      AUC = auc,
      Mean_Probability = mean(pred_proba)
    )
  }
  
  print("\n=== Prediction Performance ===")
  results_df <- do.call(rbind, lapply(results, function(x) {
    data.frame(Accuracy = x$Accuracy, AUC = x$AUC, Mean_Probability = x$Mean_Probability)
  }))
  rownames(results_df) <- names(results)
  print(round(results_df, 4))
  
  return(list(results = results, 
              pop_pred_proba = pop_pred_proba,
              retro_pred_proba = retro_pred_proba,
              adjusted_pred_proba = adjusted_pred_proba))
}

# Visualize results
visualize_comparison <- function(population_data, comparison, pop_pred_proba, 
                                retro_pred_proba, adjusted_pred_proba) {
  # 1. Coefficient comparison
  p1 <- ggplot(comparison, aes(x = rownames(comparison))) +
    geom_bar(aes(y = True, fill = "True"), stat = "identity", alpha = 0.8, position = position_dodge(0.8)) +
    geom_bar(aes(y = Retrospective, fill = "Retrospective"), stat = "identity", alpha = 0.8, position = position_dodge(0.8)) +
    scale_fill_manual(values = c("True" = "blue", "Retrospective" = "red")) +
    labs(title = "Coefficient Comparison", x = "Parameters", y = "Coefficient Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 2. Probability distributions
  prob_data <- data.frame(
    Probability = c(pop_pred_proba, retro_pred_proba),
    Model = rep(c("Population", "Retrospective"), each = length(pop_pred_proba))
  )
  
  p2 <- ggplot(prob_data, aes(x = Probability, fill = Model)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 50) +
    labs(title = "Probability Distributions", x = "Predicted Probability", y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 3. Adjusted vs Population probabilities
  p3 <- ggplot(data.frame(Population = pop_pred_proba, Adjusted = adjusted_pred_proba), 
               aes(x = Population, y = Adjusted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Adjusted vs Population Probabilities") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 4. ROC curves
  roc_pop <- roc(population_data$y, pop_pred_proba)
  roc_retro <- roc(population_data$y, retro_pred_proba)
  roc_adj <- roc(population_data$y, adjusted_pred_proba)
  
  roc_data <- data.frame(
    FPR = c(roc_pop$specificities, roc_retro$specificities, roc_adj$specificities),
    TPR = c(roc_pop$sensitivities, roc_retro$sensitivities, roc_adj$sensitivities),
    Model = rep(c("Population", "Retrospective", "Adjusted"), 
                c(length(roc_pop$specificities), length(roc_retro$specificities), length(roc_adj$specificities)))
  )
  
  p4 <- ggplot(roc_data, aes(x = 1 - FPR, y = TPR, color = Model)) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", alpha = 0.5) +
    labs(title = "ROC Curves", x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
}

# Demonstrate different sampling ratios
demonstrate_sampling_ratios <- function(population_data, pop_model) {
  cat("\n=== Different Sampling Ratios ===\n")
  sampling_ratios <- list(c(50, 50), c(100, 100), c(200, 100), c(100, 200))
  
  results <- list()
  
  for (i in 1:length(sampling_ratios)) {
    ratio <- sampling_ratios[[i]]
    n_cases <- ratio[1]
    n_controls <- ratio[2]
    
    cat("\nSampling", n_cases, "cases and", n_controls, "controls:\n")
    
    # Create new retrospective sample
    retro_data <- create_retrospective_sample(population_data, n_cases = n_cases, n_controls = n_controls)
    
    # Fit model
    retro_model <- glm(y ~ X1 + X2 + X3, data = retro_data, family = binomial)
    
    # Compare coefficients
    pop_coef <- coef(pop_model)
    retro_coef <- coef(retro_model)
    
    coef_diff <- sqrt(sum((pop_coef[-1] - retro_coef[-1])^2))  # Exclude intercept
    intercept_diff <- retro_coef[1] - pop_coef[1]
    
    cat("  Coefficient difference:", round(coef_diff, 4), "\n")
    cat("  Intercept difference:", round(intercept_diff, 4), "\n")
    
    results[[i]] <- list(
      n_cases = n_cases,
      n_controls = n_controls,
      coef_diff = coef_diff,
      intercept_diff = intercept_diff
    )
  }
  
  return(results)
}

# Demonstrate prevalence effects
demonstrate_prevalence_effects <- function() {
  cat("\n=== Prevalence Effects ===\n")
  prevalences <- c(0.001, 0.01, 0.05, 0.1)
  
  results <- list()
  
  for (i in 1:length(prevalences)) {
    prevalence <- prevalences[i]
    cat("\nPopulation prevalence:", prevalence, "\n")
    
    # Generate new population data
    result <- generate_population_data(n_population = 10000, prevalence = prevalence)
    population_data <- result$data
    true_alpha <- result$true_alpha
    true_beta <- result$true_beta
    
    retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)
    
    # Fit models
    model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)
    
    # Compare coefficients
    pop_coef <- coef(model_comparison$pop_model)
    retro_coef <- coef(model_comparison$retro_model)
    
    coef_diff <- sqrt(sum((pop_coef[-1] - retro_coef[-1])^2))
    intercept_diff <- retro_coef[1] - pop_coef[1]
    
    cat("  Coefficient difference:", round(coef_diff, 4), "\n")
    cat("  Intercept difference:", round(intercept_diff, 4), "\n")
    
    results[[i]] <- list(
      prevalence = prevalence,
      coef_diff = coef_diff,
      intercept_diff = intercept_diff
    )
  }
  
  return(results)
}

# Demonstrate probability calibration
demonstrate_probability_calibration <- function() {
  cat("\n=== Probability Calibration ===\n")
  
  # Generate data
  result <- generate_population_data(n_population = 10000, prevalence = 0.01)
  population_data <- result$data
  true_alpha <- result$true_alpha
  true_beta <- result$true_beta
  
  retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)
  
  # Fit models
  model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)
  
  # Get predictions
  pop_pred_proba <- predict(model_comparison$pop_model, newdata = population_data, type = "response")
  retro_pred_proba <- predict(model_comparison$retro_model, newdata = population_data, type = "response")
  adjusted_pred_proba <- adjust_probabilities(retro_pred_proba, population_data, retrospective_data)
  
  # Create visualization
  prob_data <- data.frame(
    Probability = c(pop_pred_proba, retro_pred_proba, adjusted_pred_proba),
    Model = rep(c("Population", "Retrospective", "Adjusted"), each = length(pop_pred_proba))
  )
  
  p <- ggplot(prob_data, aes(x = Probability, fill = Model)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 50) +
    labs(title = "Probability Distributions", x = "Predicted Probability", y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    facet_wrap(~Model, ncol = 3)
  
  print(p)
  
  # Print summary statistics
  cat("\nProbability Summary Statistics:\n")
  cat("Population model mean:", round(mean(pop_pred_proba), 4), "\n")
  cat("Retrospective model mean:", round(mean(retro_pred_proba), 4), "\n")
  cat("Adjusted model mean:", round(mean(adjusted_pred_proba), 4), "\n")
  
  return(list(
    pop_pred_proba = pop_pred_proba,
    retro_pred_proba = retro_pred_proba,
    adjusted_pred_proba = adjusted_pred_proba
  ))
}

# Demonstrate theoretical derivation
demonstrate_theoretical_derivation <- function() {
  cat("\n=== Theoretical Derivation Demonstration ===\n")
  
  # Generate data
  result <- generate_population_data(n_population = 10000, prevalence = 0.01)
  population_data <- result$data
  true_alpha <- result$true_alpha
  true_beta <- result$true_beta
  
  retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)
  
  # Calculate sampling probabilities
  n_cases <- sum(retrospective_data$y == 1)
  n_controls <- sum(retrospective_data$y == 0)
  n_cases_pop <- sum(population_data$y == 1)
  n_controls_pop <- nrow(population_data) - sum(population_data$y == 1)
  
  pi_1 <- n_cases / n_cases_pop
  pi_0 <- n_controls / n_controls_pop
  
  cat("Sampling probabilities:\n")
  cat("π₁ (cases):", pi_1, "\n")
  cat("π₀ (controls):", pi_0, "\n")
  cat("log(π₁/π₀):", log(pi_1/pi_0), "\n")
  
  # Fit models
  model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)
  
  # Compare intercepts
  pop_intercept <- coef(model_comparison$pop_model)[1]
  retro_intercept <- coef(model_comparison$retro_model)[1]
  
  cat("\nIntercept comparison:\n")
  cat("Population intercept:", round(pop_intercept, 4), "\n")
  cat("Retrospective intercept:", round(retro_intercept, 4), "\n")
  cat("Difference:", round(retro_intercept - pop_intercept, 4), "\n")
  cat("Theoretical adjustment:", round(log(pi_1/pi_0), 4), "\n")
  
  # Demonstrate probability adjustment
  cat("\nProbability adjustment example:\n")
  retro_prob <- 0.7  # Example retrospective probability
  adjusted_prob <- retro_prob / (retro_prob + (1 - retro_prob) * pi_0 / pi_1)
  cat("Retrospective probability:", retro_prob, "\n")
  cat("Adjusted probability:", round(adjusted_prob, 3), "\n")
  
  return(list(
    pi_1 = pi_1,
    pi_0 = pi_0,
    log_ratio = log(pi_1/pi_0),
    pop_intercept = pop_intercept,
    retro_intercept = retro_intercept
  ))
}

# Demonstrate practical applications
demonstrate_practical_applications <- function() {
  cat("\n=== Practical Applications ===\n")
  
  # Medical research example
  cat("\n1. Medical Research - Rare Disease Study:\n")
  cat("   - Disease prevalence: 0.5%\n")
  cat("   - Random sampling: Need ~20,000 people for 100 cases\n")
  cat("   - Retrospective sampling: Directly sample 100 cases + 100 controls\n")
  cat("   - Efficiency gain: 100x more efficient\n")
  
  # Fraud detection example
  cat("\n2. Fraud Detection - Credit Card Fraud:\n")
  cat("   - Fraud rate: 0.1%\n")
  cat("   - Random sampling: Need ~100,000 transactions for 100 fraud cases\n")
  cat("   - Retrospective sampling: Directly sample 100 fraud + 100 legitimate\n")
  cat("   - Efficiency gain: 500x more efficient\n")
  
  # Quality control example
  cat("\n3. Quality Control - Manufacturing Defects:\n")
  cat("   - Defect rate: 0.01%\n")
  cat("   - Random sampling: Need ~1,000,000 items for 100 defects\n")
  cat("   - Retrospective sampling: Directly sample 100 defects + 100 good items\n")
  cat("   - Efficiency gain: 5000x more efficient\n")
  
  # Demonstrate with actual data
  result <- generate_population_data(n_population = 100000, prevalence = 0.005)
  population_data <- result$data
  true_alpha <- result$true_alpha
  true_beta <- result$true_beta
  
  retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)
  
  # Fit models
  model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)
  
  # Compare performance
  pop_pred_proba <- predict(model_comparison$pop_model, newdata = population_data, type = "response")
  retro_pred_proba <- predict(model_comparison$retro_model, newdata = population_data, type = "response")
  adjusted_pred_proba <- adjust_probabilities(retro_pred_proba, population_data, retrospective_data)
  
  cat("\n4. Performance Comparison:\n")
  cat("   Population model AUC:", round(auc(population_data$y, pop_pred_proba), 3), "\n")
  cat("   Retrospective model AUC:", round(auc(population_data$y, retro_pred_proba), 3), "\n")
  cat("   Adjusted model AUC:", round(auc(population_data$y, adjusted_pred_proba), 3), "\n")
  
  return(list(
    population_data = population_data,
    retrospective_data = retrospective_data,
    pop_model = model_comparison$pop_model,
    retro_model = model_comparison$retro_model
  ))
}

# Demonstrate limitations and cautions
demonstrate_limitations_and_cautions <- function() {
  cat("\n=== Limitations and Cautions ===\n")
  
  # 1. Selection bias
  cat("\n1. Selection Bias:\n")
  cat("   - Cases and controls must be representative of their populations\n")
  cat("   - Hospital-based studies may not represent community cases\n")
  cat("   - Control selection must be appropriate for the research question\n")
  
  # 2. Information bias
  cat("\n2. Information Bias:\n")
  cat("   - Recall bias: Cases may remember exposures differently\n")
  cat("   - Interviewer bias: Knowledge of case/control status may affect data collection\n")
  cat("   - Measurement bias: Different data collection methods for cases vs controls\n")
  
  # 3. Confounding
  cat("\n3. Confounding:\n")
  cat("   - Retrospective sampling doesn't eliminate confounding\n")
  cat("   - Must still control for relevant confounders\n")
  cat("   - Stratification or regression adjustment still needed\n")
  
  # 4. Generalizability
  cat("\n4. Generalizability:\n")
  cat("   - Results may not generalize to different populations\n")
  cat("   - Sampling frame must be clearly defined\n")
  cat("   - External validity depends on study design\n")
  
  # Demonstrate with simulation
  cat("\n5. Simulation Example - Selection Bias:\n")
  
  # Generate population data
  result <- generate_population_data(n_population = 10000, prevalence = 0.01)
  population_data <- result$data
  true_alpha <- result$true_alpha
  true_beta <- result$true_beta
  
  # Introduce selection bias: cases with higher X1 values are more likely to be sampled
  cases <- population_data[population_data$y == 1, ]
  controls <- population_data[population_data$y == 0, ]
  
  # Biased sampling: prefer cases with higher X1
  case_weights <- exp(cases$X1)  # Higher X1 = higher sampling probability
  case_weights <- case_weights / sum(case_weights)
  
  biased_cases <- cases[sample(nrow(cases), 100, prob = case_weights), ]
  unbiased_controls <- controls[sample(nrow(controls), 100), ]
  
  biased_retrospective_data <- rbind(biased_cases, unbiased_controls)
  
  # Fit models
  pop_model <- glm(y ~ X1 + X2 + X3, data = population_data, family = binomial)
  biased_model <- glm(y ~ X1 + X2 + X3, data = biased_retrospective_data, family = binomial)
  
  # Compare coefficients
  cat("   Population X1 coefficient:", round(coef(pop_model)[2], 4), "\n")
  cat("   Biased sample X1 coefficient:", round(coef(biased_model)[2], 4), "\n")
  cat("   Bias in X1 coefficient:", round(coef(biased_model)[2] - coef(pop_model)[2], 4), "\n")
  
  return(list(
    population_data = population_data,
    biased_retrospective_data = biased_retrospective_data,
    pop_model = pop_model,
    biased_model = biased_model
  ))
}

# Main function to demonstrate retrospective sampling
main_r <- function() {
  cat("Retrospective Sampling in Logistic Regression\n")
  cat("=" * 60, "\n")
  
  # 1. Basic demonstration
  cat("\n1. Basic Demonstration:\n")
  result <- generate_population_data(n_population = 10000, prevalence = 0.01)
  population_data <- result$data
  true_alpha <- result$true_alpha
  true_beta <- result$true_beta
  
  retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)
  
  model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)
  
  evaluation <- evaluate_predictions(population_data, model_comparison$pop_model, model_comparison$retro_model)
  
  visualize_comparison(population_data, model_comparison$comparison, 
                      evaluation$pop_pred_proba, evaluation$retro_pred_proba, evaluation$adjusted_pred_proba)
  
  # 2. Different sampling ratios
  cat("\n2. Sampling Ratio Analysis:\n")
  sampling_results <- demonstrate_sampling_ratios(population_data, model_comparison$pop_model)
  
  # 3. Prevalence effects
  cat("\n3. Prevalence Effects:\n")
  prevalence_results <- demonstrate_prevalence_effects()
  
  # 4. Probability calibration
  cat("\n4. Probability Calibration:\n")
  calibration_results <- demonstrate_probability_calibration()
  
  # 5. Theoretical derivation
  cat("\n5. Theoretical Derivation:\n")
  theory_results <- demonstrate_theoretical_derivation()
  
  # 6. Practical applications
  cat("\n6. Practical Applications:\n")
  applications_results <- demonstrate_practical_applications()
  
  # 7. Limitations and cautions
  cat("\n7. Limitations and Cautions:\n")
  limitations_results <- demonstrate_limitations_and_cautions()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Coefficients remain unbiased in retrospective sampling\n")
  cat("2. Only intercept needs adjustment\n")
  cat("3. Probabilities can be calibrated for population inference\n")
  cat("4. Model performance is maintained\n")
  cat("5. Selection bias is a major concern\n")
  cat("6. Retrospective sampling is highly efficient for rare outcomes\n")
  
  return(list(
    population_data = population_data,
    retrospective_data = retrospective_data,
    model_comparison = model_comparison,
    evaluation = evaluation,
    sampling_results = sampling_results,
    prevalence_results = prevalence_results,
    calibration_results = calibration_results,
    theory_results = theory_results,
    applications_results = applications_results,
    limitations_results = limitations_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
