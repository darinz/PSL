# Loan Example - Decision Tree Classification
#
# This script demonstrates decision tree classification for loan applications,
# including feature engineering, tree construction, evaluation, and visualization.

library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(gridExtra)

# Set random seed for reproducibility
set.seed(42)

LoanDecisionTree <- setRefClass("LoanDecisionTree",
  fields = list(
    random_state = "numeric",
    tree = "rpart",
    label_encoders = "list"
  ),
  methods = list(
    
    initialize = function(random_state = 42) {
      random_state <<- random_state
      set.seed(random_state)
    },
    
    create_loan_dataset = function(n_samples = 1000) {
      # Create a synthetic loan dataset
      
      # Generate features
      credit_history <- sample(c("excellent", "good", "fair", "poor"), 
                              n_samples, replace = TRUE, 
                              prob = c(0.3, 0.4, 0.2, 0.1))
      
      income <- sample(c("high", "medium", "low"), 
                      n_samples, replace = TRUE, 
                      prob = c(0.4, 0.4, 0.2))
      
      loan_term <- sample(c("3_years", "5_years", "10_years"), 
                         n_samples, replace = TRUE, 
                         prob = c(0.3, 0.4, 0.3))
      
      age <- round(rnorm(n_samples, mean = 35, sd = 10))
      
      marital_status <- sample(c("single", "married", "divorced"), 
                              n_samples, replace = TRUE, 
                              prob = c(0.4, 0.5, 0.1))
      
      employment_years <- round(rexp(n_samples, rate = 1/3))
      
      loan_amount <- round(rlnorm(n_samples, meanlog = 10, sdlog = 0.5))
      
      # Create data frame
      df <- data.frame(
        credit_history = credit_history,
        income = income,
        loan_term = loan_term,
        age = age,
        marital_status = marital_status,
        employment_years = employment_years,
        loan_amount = loan_amount
      )
      
      # Create target variable based on business rules
      determine_loan_status <- function(row) {
        # Excellent credit - always safe
        if (row$credit_history == "excellent") {
          return("safe")
        }
        
        # Poor credit with low income - risky
        if (row$credit_history == "poor" && row$income == "low") {
          return("risky")
        }
        
        # Fair credit with short term - risky
        if (row$credit_history == "fair" && row$loan_term == "3_years") {
          return("risky")
        }
        
        # Poor credit with high income and long term - safe
        if (row$credit_history == "poor" && row$income == "high" && row$loan_term == "10_years") {
          return("safe")
        }
        
        # Good credit - generally safe
        if (row$credit_history == "good") {
          return("safe")
        }
        
        # Default case
        return("risky")
      }
      
      # Apply the function to each row
      df$loan_status <- apply(df, 1, determine_loan_status)
      
      # Add some noise to make it more realistic
      noise_indices <- sample(1:n_samples, round(0.1 * n_samples))
      df$loan_status[noise_indices] <- sample(c("safe", "risky"), 
                                             length(noise_indices), replace = TRUE)
      
      return(df)
    },
    
    preprocess_data = function(df) {
      # Preprocess the loan dataset
      df_processed <- df
      
      # Convert categorical variables to factors
      categorical_features <- c("credit_history", "income", "loan_term", "marital_status", "loan_status")
      
      for (feature in categorical_features) {
        df_processed[[feature]] <- as.factor(df_processed[[feature]])
      }
      
      return(df_processed)
    },
    
    train_tree = function(X, y, max_depth = 5, min_split = 10) {
      # Train the decision tree
      formula <- as.formula(paste("loan_status ~", paste(names(X), collapse = " + ")))
      
      tree <<- rpart(formula, data = cbind(X, loan_status = y),
                    control = rpart.control(maxdepth = max_depth, 
                                          minsplit = min_split))
    },
    
    evaluate_tree = function(X_test, y_test) {
      # Evaluate the decision tree performance
      if (is.null(tree)) {
        stop("Tree not trained yet. Call train_tree() first.")
      }
      
      # Make predictions
      y_pred <- predict(tree, X_test, type = "class")
      
      # Calculate metrics
      accuracy <- mean(y_pred == y_test)
      
      # Cross-validation score
      cv_results <- train(loan_status ~ ., data = cbind(X_test, loan_status = y_test),
                         method = "rpart",
                         trControl = trainControl(method = "cv", number = 5))
      
      return(list(
        accuracy = accuracy,
        cv_mean = cv_results$results$Accuracy[1],
        cv_std = cv_results$results$AccuracySD[1],
        predictions = y_pred
      ))
    },
    
    analyze_feature_importance = function() {
      # Analyze feature importance
      if (is.null(tree)) {
        stop("Tree not trained yet. Call train_tree() first.")
      }
      
      # Extract variable importance
      importance <- tree$variable.importance
      
      if (length(importance) > 0) {
        importance_df <- data.frame(
          feature = names(importance),
          importance = as.numeric(importance)
        )
        importance_df <- importance_df[order(-importance_df$importance), ]
        return(importance_df)
      } else {
        return(data.frame(feature = character(), importance = numeric()))
      }
    },
    
    visualize_tree = function() {
      # Visualize the decision tree
      if (is.null(tree)) {
        stop("Tree not trained yet. Call train_tree() first.")
      }
      
      # Create tree plot
      png("loan_decision_tree.png", width = 1200, height = 800, res = 150)
      rpart.plot(tree, 
                type = 4, 
                extra = 101, 
                digits = 3,
                fallen.leaves = TRUE,
                shadow.col = "gray",
                box.palette = "RdYlGn")
      dev.off()
      
      cat("Decision tree visualization saved as 'loan_decision_tree.png'\n")
    },
    
    print_tree_text = function() {
      # Print the decision tree in text format
      if (is.null(tree)) {
        stop("Tree not trained yet. Call train_tree() first.")
      }
      
      cat("Decision Tree Structure:\n")
      print(tree)
    },
    
    demonstrate_loan_scoring = function(df, sample_applications) {
      # Demonstrate loan scoring for specific applications
      cat("\n=== Loan Application Scoring Examples ===\n")
      
      for (i in seq_along(sample_applications)) {
        application <- sample_applications[[i]]
        cat(sprintf("\nApplication %d:\n", i))
        
        for (feature in names(application)) {
          cat(sprintf("  %s: %s\n", feature, application[[feature]]))
        }
        
        # Create data frame for prediction
        app_df <- data.frame(t(application))
        
        # Make prediction
        prediction <- predict(tree, app_df, type = "class")
        probability <- predict(tree, app_df, type = "prob")
        
        cat(sprintf("  Prediction: %s\n", as.character(prediction)))
        cat(sprintf("  Confidence: %.2f\n", max(probability)))
      }
    },
    
    analyze_decision_paths = function(df) {
      # Analyze decision paths for different credit histories
      cat("\n=== Decision Path Analysis ===\n")
      
      # Group by credit history
      credit_groups <- split(df, df$credit_history)
      
      for (credit_type in names(credit_groups)) {
        group <- credit_groups[[credit_type]]
        cat(sprintf("\nCredit History: %s\n", credit_type))
        cat(sprintf("Number of applications: %d\n", nrow(group)))
        
        # Calculate approval rate
        approval_rate <- mean(group$loan_status == "safe")
        cat(sprintf("Approval rate: %.2f%%\n", approval_rate * 100))
        
        # Show distribution of other features
        cat("Income distribution:\n")
        income_dist <- table(group$income)
        for (income in names(income_dist)) {
          cat(sprintf("  %s: %d\n", income, income_dist[income]))
        }
      }
    },
    
    create_loan_visualizations = function(df) {
      # Create visualizations for loan data analysis
      cat("\n=== Creating Loan Data Visualizations ===\n")
      
      # 1. Credit History vs Loan Status
      p1 <- ggplot(df, aes(x = credit_history, fill = loan_status)) +
        geom_bar(position = "dodge") +
        labs(title = "Credit History vs Loan Status", x = "Credit History", y = "Count") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      # 2. Income vs Loan Status
      p2 <- ggplot(df, aes(x = income, fill = loan_status)) +
        geom_bar(position = "dodge") +
        labs(title = "Income vs Loan Status", x = "Income", y = "Count") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      # 3. Loan Term vs Loan Status
      p3 <- ggplot(df, aes(x = loan_term, fill = loan_status)) +
        geom_bar(position = "dodge") +
        labs(title = "Loan Term vs Loan Status", x = "Loan Term", y = "Count") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      # 4. Age distribution by loan status
      p4 <- ggplot(df, aes(x = loan_status, y = age, fill = loan_status)) +
        geom_boxplot() +
        labs(title = "Age Distribution by Loan Status", x = "Loan Status", y = "Age") +
        theme_minimal()
      
      # 5. Employment years by loan status
      p5 <- ggplot(df, aes(x = loan_status, y = employment_years, fill = loan_status)) +
        geom_boxplot() +
        labs(title = "Employment Years by Loan Status", x = "Loan Status", y = "Employment Years") +
        theme_minimal()
      
      # 6. Loan amount by loan status
      p6 <- ggplot(df, aes(x = loan_status, y = loan_amount, fill = loan_status)) +
        geom_boxplot() +
        labs(title = "Loan Amount by Loan Status", x = "Loan Status", y = "Loan Amount") +
        theme_minimal()
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3)
      
      # Save plot
      ggsave("loan_analysis.png", combined_plot, width = 15, height = 10, dpi = 300)
      cat("Loan analysis visualizations saved as 'loan_analysis.png'\n")
      
      return(combined_plot)
    },
    
    run_complete_analysis = function() {
      # Run complete loan analysis
      cat("=== Loan Decision Tree Analysis ===\n")
      
      # 1. Create dataset
      cat("\n1. Creating loan dataset...\n")
      df <- create_loan_dataset(n_samples = 1000)
      cat(sprintf("Dataset created with %d samples\n", nrow(df)))
      cat("Loan status distribution:\n")
      print(table(df$loan_status))
      
      # 2. Preprocess data
      cat("\n2. Preprocessing data...\n")
      df_processed <- preprocess_data(df)
      
      # 3. Split data
      set.seed(random_state)
      train_indices <- sample(1:nrow(df_processed), 0.7 * nrow(df_processed))
      train_data <- df_processed[train_indices, ]
      test_data <- df_processed[-train_indices, ]
      
      X_train <- train_data[, !names(train_data) %in% "loan_status"]
      y_train <- train_data$loan_status
      X_test <- test_data[, !names(test_data) %in% "loan_status"]
      y_test <- test_data$loan_status
      
      # 4. Train tree
      cat("\n3. Training decision tree...\n")
      train_tree(X_train, y_train, max_depth = 5)
      
      # 5. Evaluate performance
      cat("\n4. Evaluating performance...\n")
      results <- evaluate_tree(X_test, y_test)
      cat(sprintf("Accuracy: %.3f\n", results$accuracy))
      cat(sprintf("Cross-validation score: %.3f (+/- %.3f)\n", 
                  results$cv_mean, results$cv_std * 2))
      
      # 6. Feature importance
      cat("\n5. Analyzing feature importance...\n")
      importance_df <- analyze_feature_importance()
      if (nrow(importance_df) > 0) {
        cat("Feature Importance:\n")
        print(importance_df)
      } else {
        cat("No variable importance available for this tree.\n")
      }
      
      # 7. Visualize tree
      cat("\n6. Creating tree visualization...\n")
      visualize_tree()
      
      # 8. Print tree structure
      cat("\n7. Tree structure in text format:\n")
      print_tree_text()
      
      # 9. Demonstrate loan scoring
      cat("\n8. Loan scoring examples...\n")
      sample_applications <- list(
        list(
          credit_history = "excellent",
          income = "high",
          loan_term = "5_years",
          age = 35,
          marital_status = "married",
          employment_years = 5,
          loan_amount = 50000
        ),
        list(
          credit_history = "poor",
          income = "low",
          loan_term = "3_years",
          age = 25,
          marital_status = "single",
          employment_years = 1,
          loan_amount = 10000
        ),
        list(
          credit_history = "fair",
          income = "medium",
          loan_term = "10_years",
          age = 45,
          marital_status = "married",
          employment_years = 10,
          loan_amount = 75000
        )
      )
      demonstrate_loan_scoring(df, sample_applications)
      
      # 10. Analyze decision paths
      cat("\n9. Decision path analysis...\n")
      analyze_decision_paths(df_processed)
      
      # 11. Create visualizations
      cat("\n10. Creating visualizations...\n")
      create_loan_visualizations(df)
      
      return(list(
        dataset = df,
        processed_data = df_processed,
        results = results,
        importance = importance_df
      ))
    }
  )
)

# Main function to run the loan analysis
main <- function() {
  # Create loan decision tree instance
  loan_tree <- LoanDecisionTree$new(random_state = 42)
  
  # Run complete analysis
  results <- loan_tree$run_complete_analysis()
  
  cat("\n=== Analysis Complete ===\n")
  cat("Check the generated files:\n")
  cat("- loan_decision_tree.png: Decision tree visualization\n")
  cat("- loan_analysis.png: Data analysis plots\n")
}

# Run the main function
main()
