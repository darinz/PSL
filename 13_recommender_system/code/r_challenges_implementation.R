# Challenges in Recommender Systems - R Implementation
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Generate synthetic data with challenges
generate_synthetic_challenge_data <- function(n_users = 1000, n_items = 500, n_ratings = 5000, seed = 42) {
  set.seed(seed)
  
  # Create synthetic ratings with challenges
  ratings_data <- list()
  
  # Create popular items and active users
  popular_items <- sample(1:n_items, 50, replace = FALSE)
  active_users <- sample(1:n_users, 100, replace = FALSE)
  
  for (user_id in 1:n_users) {
    # Vary number of ratings based on user activity
    if (user_id %in% active_users) {
      n_user_ratings <- sample(20:50, 1)
    } else {
      n_user_ratings <- sample(1:10, 1)
    }
    
    rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
    
    for (item_id in rated_items) {
      # Create popularity bias
      if (item_id %in% popular_items) {
        base_rating <- rnorm(1, 4.0, 0.5)
      } else {
        base_rating <- rnorm(1, 3.0, 0.8)
      }
      
      # Add cold start users
      if (runif(1) < 0.1) {
        base_rating <- rnorm(1, 3.0, 1.0)
      }
      
      rating <- max(1, min(5, base_rating))
      
      ratings_data[[length(ratings_data) + 1]] <- list(
        user_id = user_id,
        item_id = item_id,
        rating = rating
      )
    }
  }
  
  ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))
  return(list(ratings_df = ratings_df, popular_items = popular_items, active_users = active_users))
}

# Calculate Gini coefficient
calculate_gini <- function(values) {
  sorted_values <- sort(values)
  n <- length(sorted_values)
  cumsum_values <- cumsum(sorted_values)
  return((n + 1 - 2 * sum(cumsum_values) / cumsum_values[n]) / n)
}

# Analyze cold start problem
analyze_cold_start <- function(ratings_df, user_col = "user_id", item_col = "item_id") {
  # Count ratings per user and item
  user_counts <- ratings_df %>%
    group_by(!!sym(user_col)) %>%
    summarise(n_ratings = n()) %>%
    arrange(desc(n_ratings))
  
  item_counts <- ratings_df %>%
    group_by(!!sym(item_col)) %>%
    summarise(n_ratings = n()) %>%
    arrange(desc(n_ratings))
  
  # Identify cold start cases
  cold_start_users <- sum(user_counts$n_ratings <= 1)
  cold_start_items <- sum(item_counts$n_ratings <= 1)
  
  # Calculate statistics
  total_users <- nrow(user_counts)
  total_items <- nrow(item_counts)
  
  cold_start_stats <- list(
    cold_start_users = cold_start_users,
    cold_start_items = cold_start_items,
    user_cold_start_rate = cold_start_users / total_users,
    item_cold_start_rate = cold_start_items / total_items,
    avg_ratings_per_user = mean(user_counts$n_ratings),
    avg_ratings_per_item = mean(item_counts$n_ratings),
    median_ratings_per_user = median(user_counts$n_ratings),
    median_ratings_per_item = median(item_counts$n_ratings)
  )
  
  return(list(stats = cold_start_stats, user_counts = user_counts, item_counts = item_counts))
}

# Analyze data sparsity
analyze_sparsity <- function(ratings_df, user_col = "user_id", item_col = "item_id") {
  # Create rating matrix
  rating_matrix <- ratings_df %>%
    spread(!!sym(item_col), rating, fill = NA) %>%
    select(-!!sym(user_col)) %>%
    as.matrix()
  
  # Calculate sparsity
  total_entries <- nrow(rating_matrix) * ncol(rating_matrix)
  observed_entries <- sum(!is.na(rating_matrix))
  sparsity <- 1 - (observed_entries / total_entries)
  
  # Analyze rating distribution
  rating_distribution <- table(ratings_df$rating)
  
  # Calculate coverage metrics
  user_coverage <- rowSums(!is.na(rating_matrix))
  item_coverage <- colSums(!is.na(rating_matrix))
  
  sparsity_stats <- list(
    sparsity = sparsity,
    total_entries = total_entries,
    observed_entries = observed_entries,
    avg_user_coverage = mean(user_coverage),
    avg_item_coverage = mean(item_coverage),
    min_user_coverage = min(user_coverage),
    max_user_coverage = max(user_coverage),
    min_item_coverage = min(item_coverage),
    max_item_coverage = max(item_coverage)
  )
  
  return(list(stats = sparsity_stats, rating_matrix = rating_matrix, rating_distribution = rating_distribution))
}

# Analyze popularity bias
analyze_popularity_bias <- function(ratings_df, user_col = "user_id", item_col = "item_id") {
  # Calculate item popularity
  item_popularity <- ratings_df %>%
    group_by(!!sym(item_col)) %>%
    summarise(n_ratings = n()) %>%
    arrange(desc(n_ratings))
  
  # Calculate user activity
  user_activity <- ratings_df %>%
    group_by(!!sym(user_col)) %>%
    summarise(n_ratings = n()) %>%
    arrange(desc(n_ratings))
  
  # Calculate popularity bias metrics
  gini_coefficient_items <- calculate_gini(item_popularity$n_ratings)
  gini_coefficient_users <- calculate_gini(user_activity$n_ratings)
  
  # Calculate recommendation diversity
  top_items <- head(item_popularity, 10)
  bottom_items <- tail(item_popularity, 10)
  
  popularity_stats <- list(
    gini_coefficient_items = gini_coefficient_items,
    gini_coefficient_users = gini_coefficient_users,
    top_10_items_share = sum(top_items$n_ratings) / sum(item_popularity$n_ratings),
    bottom_10_items_share = sum(bottom_items$n_ratings) / sum(item_popularity$n_ratings),
    popularity_ratio = max(item_popularity$n_ratings) / min(item_popularity$n_ratings),
    activity_ratio = max(user_activity$n_ratings) / min(user_activity$n_ratings)
  )
  
  return(list(stats = popularity_stats, item_popularity = item_popularity, user_activity = user_activity))
}

# Analyze scalability challenges
analyze_scalability <- function(ratings_df, user_col = "user_id", item_col = "item_id") {
  n_users <- length(unique(ratings_df[[user_col]]))
  n_items <- length(unique(ratings_df[[item_col]]))
  n_ratings <- nrow(ratings_df)
  
  # Calculate computational complexity estimates
  ubcf_complexity <- n_users^2 * n_items
  ibcf_complexity <- n_items^2 * n_users
  mf_complexity <- n_ratings * 10 * 100  # Assuming 10 factors, 100 epochs
  
  # Memory requirements (8 bytes per float)
  user_sim_memory <- n_users^2 * 8
  item_sim_memory <- n_items^2 * 8
  
  scalability_stats <- list(
    n_users = n_users,
    n_items = n_items,
    n_ratings = n_ratings,
    ubcf_complexity = ubcf_complexity,
    ibcf_complexity = ibcf_complexity,
    mf_complexity = mf_complexity,
    user_sim_memory_mb = user_sim_memory / (1024 * 1024),
    item_sim_memory_mb = item_sim_memory / (1024 * 1024),
    user_item_ratio = n_users / n_items,
    density = n_ratings / (n_users * n_items)
  )
  
  return(scalability_stats)
}

# Simulate cold start impact
simulate_cold_start_impact <- function(ratings_df, user_col = "user_id", item_col = "item_id", 
                                     rating_col = "rating", test_fraction = 0.1) {
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), (1 - test_fraction) * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Identify cold start cases in test set
  train_users <- unique(train_df[[user_col]])
  train_items <- unique(train_df[[item_col]])
  
  cold_start_test <- test_df[
    (!test_df[[user_col]] %in% train_users) | 
    (!test_df[[item_col]] %in% train_items), 
  ]
  
  regular_test <- test_df[
    (test_df[[user_col]] %in% train_users) & 
    (test_df[[item_col]] %in% train_items), 
  ]
  
  # Calculate baseline predictions
  global_mean <- mean(train_df[[rating_col]])
  
  # Evaluate on different test sets
  if (nrow(cold_start_test) > 0) {
    cold_start_mae <- mean(abs(cold_start_test[[rating_col]] - global_mean))
  } else {
    cold_start_mae <- 0
  }
  
  if (nrow(regular_test) > 0) {
    regular_mae <- mean(abs(regular_test[[rating_col]] - global_mean))
  } else {
    regular_mae <- 0
  }
  
  impact_stats <- list(
    cold_start_mae = cold_start_mae,
    regular_mae = regular_mae,
    cold_start_ratio = nrow(cold_start_test) / nrow(test_df),
    performance_degradation = ifelse(regular_mae > 0, cold_start_mae / regular_mae, Inf)
  )
  
  return(list(stats = impact_stats, cold_test = cold_start_test, regular_test = regular_test))
}

# Analyze bias mitigation strategies
analyze_bias_mitigation <- function(ratings_df, user_col = "user_id", item_col = "item_id") {
  # Calculate item popularity
  item_popularity <- ratings_df %>%
    group_by(!!sym(item_col)) %>%
    summarise(n_ratings = n()) %>%
    arrange(desc(n_ratings))
  
  # Calculate popularity bias
  popularity_bias <- item_popularity$n_ratings / sum(item_popularity$n_ratings)
  
  # Apply debiasing techniques
  # 1. Inverse popularity sampling
  inverse_popularity <- 1 / (item_popularity$n_ratings + 1)  # Add 1 to avoid division by zero
  debiased_popularity <- inverse_popularity / sum(inverse_popularity)
  
  # 2. Square root debiasing
  sqrt_popularity <- sqrt(item_popularity$n_ratings)
  sqrt_debiased <- sqrt_popularity / sum(sqrt_popularity)
  
  # 3. Log debiasing
  log_popularity <- log(item_popularity$n_ratings + 1)
  log_debiased <- log_popularity / sum(log_popularity)
  
  bias_mitigation_stats <- list(
    original_gini = calculate_gini(item_popularity$n_ratings),
    inverse_gini = calculate_gini(debiased_popularity * sum(item_popularity$n_ratings)),
    sqrt_gini = calculate_gini(sqrt_debiased * sum(item_popularity$n_ratings)),
    log_gini = calculate_gini(log_debiased * sum(item_popularity$n_ratings)),
    popularity_correlation = cor(item_popularity$n_ratings, 1:nrow(item_popularity))
  )
  
  return(list(
    stats = bias_mitigation_stats,
    distributions = list(
      original = popularity_bias,
      inverse = debiased_popularity,
      sqrt = sqrt_debiased,
      log = log_debiased
    )
  ))
}

# Demonstrate cold start analysis
demonstrate_cold_start_analysis <- function() {
  cat("=== Cold Start Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Analyze cold start
  cold_start_result <- analyze_cold_start(ratings_df)
  
  for (key in names(cold_start_result$stats)) {
    cat(sprintf("%s: %.4f\n", key, cold_start_result$stats[[key]]))
  }
  
  return(cold_start_result)
}

# Demonstrate sparsity analysis
demonstrate_sparsity_analysis <- function() {
  cat("=== Sparsity Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Analyze sparsity
  sparsity_result <- analyze_sparsity(ratings_df)
  
  for (key in names(sparsity_result$stats)) {
    cat(sprintf("%s: %.4f\n", key, sparsity_result$stats[[key]]))
  }
  
  return(sparsity_result)
}

# Demonstrate popularity bias analysis
demonstrate_popularity_bias_analysis <- function() {
  cat("=== Popularity Bias Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Analyze popularity bias
  popularity_result <- analyze_popularity_bias(ratings_df)
  
  for (key in names(popularity_result$stats)) {
    cat(sprintf("%s: %.4f\n", key, popularity_result$stats[[key]]))
  }
  
  return(popularity_result)
}

# Demonstrate scalability analysis
demonstrate_scalability_analysis <- function() {
  cat("=== Scalability Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Analyze scalability
  scalability_stats <- analyze_scalability(ratings_df)
  
  for (key in names(scalability_stats)) {
    cat(sprintf("%s: %.2f\n", key, scalability_stats[[key]]))
  }
  
  return(scalability_stats)
}

# Demonstrate cold start impact simulation
demonstrate_cold_start_impact <- function() {
  cat("=== Cold Start Impact Simulation ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Simulate cold start impact
  impact_result <- simulate_cold_start_impact(ratings_df)
  
  for (key in names(impact_result$stats)) {
    cat(sprintf("%s: %.4f\n", key, impact_result$stats[[key]]))
  }
  
  return(impact_result)
}

# Demonstrate bias mitigation analysis
demonstrate_bias_mitigation <- function() {
  cat("=== Bias Mitigation Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  # Analyze bias mitigation
  bias_result <- analyze_bias_mitigation(ratings_df)
  
  for (key in names(bias_result$stats)) {
    cat(sprintf("%s: %.4f\n", key, bias_result$stats[[key]]))
  }
  
  return(bias_result)
}

# Demonstrate visualization
demonstrate_visualization <- function() {
  cat("=== Challenge Visualization ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  popular_items <- data_result$popular_items
  active_users <- data_result$active_users
  
  # Analyze all challenges
  cold_start_result <- analyze_cold_start(ratings_df)
  sparsity_result <- analyze_sparsity(ratings_df)
  popularity_result <- analyze_popularity_bias(ratings_df)
  scalability_stats <- analyze_scalability(ratings_df)
  impact_result <- simulate_cold_start_impact(ratings_df)
  bias_result <- analyze_bias_mitigation(ratings_df)
  
  # Visualization
  # Cold start analysis
  p1 <- ggplot(cold_start_result$user_counts, aes(x = n_ratings)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
    labs(title = "User Rating Distribution",
         x = "Number of Ratings", y = "Frequency") +
    theme_minimal()
  
  p2 <- ggplot(cold_start_result$item_counts, aes(x = n_ratings)) +
    geom_histogram(bins = 30, fill = "lightcoral", alpha = 0.7) +
    labs(title = "Item Rating Distribution",
         x = "Number of Ratings", y = "Frequency") +
    theme_minimal()
  
  # Popularity bias
  p3 <- ggplot(head(popularity_result$item_popularity, 20), 
               aes(x = reorder(factor(item_id), n_ratings), y = n_ratings)) +
    geom_bar(stat = "identity", fill = "orange") +
    labs(title = "Top 20 Most Popular Items",
         x = "Item ID", y = "Number of Ratings") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Rating distribution
  p4 <- ggplot(ratings_df, aes(x = factor(rating))) +
    geom_bar(fill = "green", alpha = 0.7) +
    labs(title = "Rating Distribution",
         x = "Rating", y = "Count") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  return(list(
    cold_start_stats = cold_start_result$stats,
    sparsity_stats = sparsity_result$stats,
    popularity_stats = popularity_result$stats,
    scalability_stats = scalability_stats,
    impact_stats = impact_result$stats,
    bias_stats = bias_result$stats
  ))
}

# Demonstrate detailed analysis
demonstrate_detailed_analysis <- function() {
  cat("=== Detailed Challenge Analysis ===\n")
  
  # Generate data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  popular_items <- data_result$popular_items
  active_users <- data_result$active_users
  
  # Cold start impact simulation
  impact_result <- simulate_cold_start_impact(ratings_df)
  
  # Cold start impact by user type
  cat("Cold Start Impact by User Type:\n")
  active_cold_start <- length(intersect(active_users, unique(impact_result$cold_test$user_id)))
  inactive_cold_start <- length(setdiff(unique(impact_result$cold_test$user_id), active_users))
  cat(sprintf("Active users in cold start: %d\n", active_cold_start))
  cat(sprintf("Inactive users in cold start: %d\n", inactive_cold_start))
  
  # Popularity bias analysis
  popularity_result <- analyze_popularity_bias(ratings_df)
  cat(sprintf("\nPopularity Bias Analysis:\n"))
  cat(sprintf("Top 10%% items account for %.2f%% of ratings\n", 
              popularity_result$stats$top_10_items_share * 100))
  cat(sprintf("Bottom 10%% items account for %.2f%% of ratings\n", 
              popularity_result$stats$bottom_10_items_share * 100))
  cat(sprintf("Popularity ratio: %.2f\n", popularity_result$stats$popularity_ratio))
  
  # Scalability recommendations
  scalability_stats <- analyze_scalability(ratings_df)
  cat(sprintf("\nScalability Recommendations:\n"))
  if (scalability_stats$user_item_ratio > 2) {
    cat("Recommend IBCF (more users than items)\n")
  } else if (scalability_stats$user_item_ratio < 0.5) {
    cat("Recommend UBCF (more items than users)\n")
  } else {
    cat("Consider both UBCF and IBCF\n")
  }
  
  if (scalability_stats$user_sim_memory_mb > 1000) {
    cat("User similarity matrix too large - consider sampling\n")
  }
  if (scalability_stats$item_sim_memory_mb > 1000) {
    cat("Item similarity matrix too large - consider sampling\n")
  }
  
  # Bias mitigation effectiveness
  bias_result <- analyze_bias_mitigation(ratings_df)
  cat(sprintf("\nBias Mitigation Effectiveness:\n"))
  improvements <- list(
    Inverse = bias_result$stats$original_gini - bias_result$stats$inverse_gini,
    Sqrt = bias_result$stats$original_gini - bias_result$stats$sqrt_gini,
    Log = bias_result$stats$original_gini - bias_result$stats$log_gini
  )
  best_method <- names(which.max(unlist(improvements)))
  cat(sprintf("Best debiasing method: %s (improvement: %.4f)\n", 
              best_method, improvements[[best_method]]))
  
  return(list(
    impact_stats = impact_result$stats,
    popularity_stats = popularity_result$stats,
    scalability_stats = scalability_stats,
    bias_stats = bias_result$stats,
    improvements = improvements
  ))
}

# Main function
main_r <- function() {
  cat("Recommender System Challenges: Comprehensive Analysis\n")
  cat("=" * 60, "\n")
  
  # Generate synthetic data
  data_result <- generate_synthetic_challenge_data()
  ratings_df <- data_result$ratings_df
  
  cat("Synthetic Dataset with Challenges:\n")
  cat(sprintf("Number of users: %d\n", length(unique(ratings_df$user_id))))
  cat(sprintf("Number of items: %d\n", length(unique(ratings_df$item_id))))
  cat(sprintf("Number of ratings: %d\n", nrow(ratings_df)))
  
  # 1. Cold start analysis
  cold_start_result <- demonstrate_cold_start_analysis()
  
  # 2. Sparsity analysis
  sparsity_result <- demonstrate_sparsity_analysis()
  
  # 3. Popularity bias analysis
  popularity_result <- demonstrate_popularity_bias_analysis()
  
  # 4. Scalability analysis
  scalability_stats <- demonstrate_scalability_analysis()
  
  # 5. Cold start impact simulation
  impact_result <- demonstrate_cold_start_impact()
  
  # 6. Bias mitigation analysis
  bias_result <- demonstrate_bias_mitigation()
  
  # 7. Comprehensive visualization
  viz_results <- demonstrate_visualization()
  
  # 8. Detailed analysis
  detailed_results <- demonstrate_detailed_analysis()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Cold start affects a significant portion of users and items\n")
  cat("2. Data sparsity is a fundamental challenge in recommendation systems\n")
  cat("3. Popularity bias creates feedback loops that reinforce existing patterns\n")
  cat("4. Scalability becomes critical as system size grows\n")
  cat("5. Bias mitigation techniques can improve fairness and diversity\n")
  cat("6. Multiple challenges often interact and compound each other\n")
  
  return(list(
    ratings_df = ratings_df,
    cold_start_stats = cold_start_result$stats,
    sparsity_stats = sparsity_result$stats,
    popularity_stats = popularity_result$stats,
    scalability_stats = scalability_stats,
    impact_stats = impact_result$stats,
    bias_stats = bias_result$stats,
    detailed_results = detailed_results
  ))
}
