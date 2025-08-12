# Content-Based Recommender System in R

# Load required libraries
library(tm)
library(proxy)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(cluster)
library(factoextra)

# Content-based recommender function
content_based_recommender <- function(similarity_metric = "cosine") {
  list(
    similarity_metric = similarity_metric,
    item_profiles = NULL,
    user_profiles = NULL,
    feature_names = NULL
  )
}

# Compute similarity between two profiles
compute_similarity <- function(profile1, profile2, metric = "cosine") {
  if (metric == "cosine") {
    return(sum(profile1 * profile2) / (sqrt(sum(profile1^2)) * sqrt(sum(profile2^2))))
  } else if (metric == "euclidean") {
    distance <- sqrt(sum((profile1 - profile2)^2))
    return(1 / (1 + distance))
  } else if (metric == "pearson") {
    return(cor(profile1, profile2, method = "pearson"))
  }
}

# Create item profiles from features
create_item_profiles <- function(recommender, items_df, feature_columns, text_columns = NULL) {
  profiles <- list()
  feature_names <- c()
  
  # Handle categorical features
  for (col in feature_columns) {
    if (is.character(items_df[[col]])) {
      # Encode categorical features
      unique_values <- unique(items_df[[col]])
      encoded_matrix <- matrix(0, nrow = nrow(items_df), ncol = length(unique_values))
      
      for (i in 1:length(unique_values)) {
        encoded_matrix[items_df[[col]] == unique_values[i], i] <- 1
      }
      
      profiles[[length(profiles) + 1]] <- encoded_matrix
      feature_names <- c(feature_names, paste0(col, "_", unique_values))
    } else {
      # Numerical features
      profiles[[length(profiles) + 1]] <- matrix(items_df[[col]], ncol = 1)
      feature_names <- c(feature_names, col)
    }
  }
  
  # Handle text features
  if (!is.null(text_columns)) {
    for (col in text_columns) {
      # Create corpus
      corpus <- Corpus(VectorSource(items_df[[col]]))
      
      # Create document-term matrix
      dtm <- DocumentTermMatrix(corpus, control = list(
        removePunctuation = TRUE,
        removeNumbers = TRUE,
        stopwords = TRUE,
        weighting = weightTfIdf
      ))
      
      # Convert to matrix
      text_matrix <- as.matrix(dtm)
      
      # Limit features
      if (ncol(text_matrix) > 50) {
        text_matrix <- text_matrix[, 1:50]
      }
      
      profiles[[length(profiles) + 1]] <- text_matrix
      feature_names <- c(feature_names, paste0(col, "_", colnames(text_matrix)))
    }
  }
  
  # Combine all features
  recommender$item_profiles <- do.call(cbind, profiles)
  recommender$feature_names <- feature_names
  
  # Normalize features
  recommender$item_profiles <- scale(recommender$item_profiles)
  
  return(recommender)
}

# Create user profiles from ratings
create_user_profiles <- function(recommender, ratings_df, items_df) {
  user_profiles <- list()
  
  for (user_id in unique(ratings_df$user_id)) {
    user_ratings <- ratings_df[ratings_df$user_id == user_id, ]
    
    # Get items rated by this user
    rated_items <- user_ratings$item_id
    ratings <- user_ratings$rating
    
    # Find corresponding item profiles
    item_indices <- match(rated_items, items_df$movie_id)
    item_profiles <- recommender$item_profiles[item_indices, ]
    
    # Compute weighted average (weighted by ratings)
    weights <- ratings / sum(ratings)
    user_profile <- colSums(t(item_profiles) * weights)
    
    user_profiles[[as.character(user_id)]] <- user_profile
  }
  
  recommender$user_profiles <- user_profiles
  return(recommender)
}

# Generate recommendations for a user
recommend <- function(recommender, user_id, n_recommendations = 5) {
  if (!(as.character(user_id) %in% names(recommender$user_profiles))) {
    return(list())
  }
  
  user_profile <- recommender$user_profiles[[as.character(user_id)]]
  
  # Compute similarities with all items
  similarities <- sapply(1:nrow(recommender$item_profiles), function(i) {
    compute_similarity(user_profile, recommender$item_profiles[i, ], recommender$similarity_metric)
  })
  
  # Sort by similarity
  sorted_indices <- order(similarities, decreasing = TRUE)
  
  # Return top recommendations
  result <- list()
  for (i in 1:n_recommendations) {
    result[[i]] <- list(
      item_index = sorted_indices[i],
      similarity = similarities[sorted_indices[i]]
    )
  }
  
  return(result)
}

# Get feature importance for a user
get_feature_importance <- function(recommender, user_id, top_features = 10) {
  if (!(as.character(user_id) %in% names(recommender$user_profiles))) {
    return(list())
  }
  
  user_profile <- recommender$user_profiles[[as.character(user_id)]]
  
  # Get feature importance (absolute values)
  feature_importance <- data.frame(
    feature = recommender$feature_names,
    importance = abs(user_profile)
  )
  
  feature_importance <- feature_importance[order(feature_importance$importance, decreasing = TRUE), ]
  
  return(feature_importance[1:top_features, ])
}

# Generate synthetic movie data
generate_synthetic_movie_data <- function(n_movies = 100, n_users = 50, seed = 42) {
  set.seed(seed)
  
  # Create movie features
  movies_df <- data.frame(
    movie_id = 1:n_movies,
    title = paste0("Movie_", 1:n_movies),
    genre = sample(c("Action", "Drama", "Comedy", "Thriller", "Romance"), n_movies, replace = TRUE),
    year = sample(1990:2023, n_movies, replace = TRUE),
    rating = runif(n_movies, 1, 10),
    budget = runif(n_movies, 1, 100),
    director = sample(c("Spielberg", "Nolan", "Tarantino", "Scorsese", "Cameron"), n_movies, replace = TRUE),
    description = paste0("Description for movie ", 1:n_movies)
  )
  
  # Create synthetic ratings
  ratings_data <- list()
  for (user_id in 1:n_users) {
    n_ratings <- sample(5:20, 1)
    rated_movies <- sample(1:n_movies, n_ratings, replace = FALSE)
    
    for (movie_id in rated_movies) {
      movie <- movies_df[movie_id, ]
      base_rating <- 5
      
      # Genre preferences
      if (movie$genre %in% c("Action", "Thriller")) {
        base_rating <- base_rating + rnorm(1, 1, 1)
      } else if (movie$genre %in% c("Drama", "Romance")) {
        base_rating <- base_rating + rnorm(1, -1, 1)
      }
      
      # Year preference
      year_factor <- (movie$year - 1990) / (2023 - 1990)
      base_rating <- base_rating + year_factor * 2
      
      # Add noise
      rating <- max(1, min(10, base_rating + rnorm(1, 0, 1)))
      
      ratings_data[[length(ratings_data) + 1]] <- list(
        user_id = user_id,
        movie_id = movie_id,
        rating = rating
      )
    }
  }
  
  ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))
  
  return(list(movies_df = movies_df, ratings_df = ratings_df))
}

# Demonstrate basic content-based recommender
demonstrate_basic_content_based <- function() {
  cat("=== Basic Content-Based Recommender System ===\n\n")
  
  # Generate synthetic data
  data <- generate_synthetic_movie_data()
  movies_df <- data$movies_df
  ratings_df <- data$ratings_df
  
  cat("Synthetic Movie Dataset:\n")
  cat("Number of movies:", nrow(movies_df), "\n")
  cat("Number of users:", length(unique(ratings_df$user_id)), "\n")
  cat("Number of ratings:", nrow(ratings_df), "\n")
  
  # Initialize and train content-based recommender
  recommender <- content_based_recommender("cosine")
  
  # Create item profiles
  feature_columns <- c("genre", "year", "rating", "budget", "director")
  text_columns <- c("description")
  
  recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
  recommender <- create_user_profiles(recommender, ratings_df, movies_df)
  
  cat("\nItem profiles shape:", dim(recommender$item_profiles), "\n")
  cat("Number of features:", length(recommender$feature_names), "\n")
  cat("Number of user profiles:", length(recommender$user_profiles), "\n")
  
  # Generate recommendations for a sample user
  test_user <- 1
  recommendations <- recommend(recommender, test_user, 10)
  
  cat("\nTop 10 recommendations for User", test_user, ":\n")
  for (i in 1:length(recommendations)) {
    item_idx <- recommendations[[i]]$item_index
    similarity <- recommendations[[i]]$similarity
    movie <- movies_df[item_idx, ]
    cat(sprintf("%d. %s (%s, %d) - Similarity: %.3f\n", 
                i, movie$title, movie$genre, movie$year, similarity))
  }
  
  return(list(movies_df = movies_df, ratings_df = ratings_df, recommender = recommender))
}

# Demonstrate feature importance analysis
demonstrate_feature_importance <- function() {
  cat("=== Feature Importance Analysis ===\n\n")
  
  # Generate data and train recommender
  data <- demonstrate_basic_content_based()
  recommender <- data$recommender
  
  # Get feature importance for multiple users
  test_users <- c(1, 2, 3)
  
  for (user_id in test_users) {
    feature_importance <- get_feature_importance(recommender, user_id, 10)
    
    cat("\nTop 10 most important features for User", user_id, ":\n")
    for (i in 1:nrow(feature_importance)) {
      cat(sprintf("  %s: %.3f\n", feature_importance$feature[i], feature_importance$importance[i]))
    }
  }
  
  # Visualize feature importance
  plots <- list()
  
  for (i in 1:length(test_users)) {
    user_id <- test_users[i]
    feature_importance <- get_feature_importance(recommender, user_id, 10)
    
    p <- ggplot(feature_importance, aes(x = importance, y = reorder(feature, importance))) +
      geom_bar(stat = "identity", fill = "steelblue") +
      labs(title = paste("Feature Importance - User", user_id),
           x = "Importance", y = "Feature") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 8))
    
    plots[[i]] <- p
  }
  
  # Combine plots
  grid.arrange(grobs = plots, ncol = 3)
  
  return(plots)
}

# Demonstrate different similarity metrics
demonstrate_similarity_metrics <- function() {
  cat("=== Similarity Metrics Comparison ===\n\n")
  
  # Generate data
  data <- generate_synthetic_movie_data()
  movies_df <- data$movies_df
  ratings_df <- data$ratings_df
  
  # Compare different similarity metrics
  similarity_metrics <- c("cosine", "euclidean", "pearson")
  results <- list()
  
  for (metric in similarity_metrics) {
    cat("Testing", toupper(metric), "similarity...\n")
    
    recommender <- content_based_recommender(metric)
    
    # Create profiles
    feature_columns <- c("genre", "year", "rating", "budget", "director")
    text_columns <- c("description")
    
    recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
    recommender <- create_user_profiles(recommender, ratings_df, movies_df)
    
    # Generate recommendations
    test_user <- 1
    recommendations <- recommend(recommender, test_user, 5)
    results[[metric]] <- recommendations
    
    cat("Top 5 recommendations:\n")
    for (i in 1:length(recommendations)) {
      item_idx <- recommendations[[i]]$item_index
      similarity <- recommendations[[i]]$similarity
      movie <- movies_df[item_idx, ]
      cat(sprintf("  %d. %s - Similarity: %.3f\n", i, movie$title, similarity))
    }
  }
  
  # Visualize similarity distributions
  plots <- list()
  
  for (i in 1:length(similarity_metrics)) {
    metric <- similarity_metrics[i]
    recommendations <- results[[metric]]
    similarities <- sapply(recommendations, function(x) x$similarity)
    
    p <- ggplot(data.frame(similarity = similarities), aes(x = similarity)) +
      geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7, color = "black") +
      labs(title = paste(toupper(metric), "Similarity Distribution"),
           x = "Similarity Score", y = "Frequency") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10))
    
    plots[[i]] <- p
  }
  
  # Combine plots
  grid.arrange(grobs = plots, ncol = 3)
  
  return(results)
}

# Demonstrate profile visualization
demonstrate_profile_visualization <- function() {
  cat("=== Profile Visualization ===\n\n")
  
  # Generate data and train recommender
  data <- demonstrate_basic_content_based()
  recommender <- data$recommender
  
  # Get user profiles for visualization
  user_ids <- c(1, 2, 3)
  user_profiles <- do.call(rbind, lapply(user_ids, function(id) {
    recommender$user_profiles[[as.character(id)]]
  }))
  
  # Apply PCA for visualization
  pca_result <- prcomp(user_profiles, scale. = TRUE)
  
  # Create visualization
  pca_df <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    User = paste("User", user_ids)
  )
  
  p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, label = User)) +
    geom_point(size = 3, color = "red") +
    geom_text(vjust = -0.5) +
    labs(title = "User Profiles in 2D Space",
         x = paste0("PC1 (", round(pca_result$sdev[1]^2 / sum(pca_result$sdev^2) * 100, 1), "% variance)"),
         y = paste0("PC2 (", round(pca_result$sdev[2]^2 / sum(pca_result$sdev^2) * 100, 1), "% variance)")) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Profile clustering
  cat("Analyzing profile clusters...\n")
  
  # Get all user profiles
  all_user_profiles <- do.call(rbind, recommender$user_profiles)
  
  # Apply K-means clustering
  kmeans_result <- kmeans(all_user_profiles, centers = 3, nstart = 25)
  
  # Visualize clusters
  cluster_df <- data.frame(
    PC1 = prcomp(all_user_profiles, scale. = TRUE)$x[, 1],
    PC2 = prcomp(all_user_profiles, scale. = TRUE)$x[, 2],
    Cluster = as.factor(kmeans_result$cluster)
  )
  
  p2 <- ggplot(cluster_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(size = 2) +
    labs(title = "User Profile Clusters",
         x = "PC1", y = "PC2") +
    theme_minimal() +
    scale_color_viridis_d()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  # Analyze cluster characteristics
  for (cluster_id in 1:3) {
    cluster_users <- which(kmeans_result$cluster == cluster_id)
    cat(sprintf("\nCluster %d has %d users\n", cluster_id, length(cluster_users)))
    
    # Get average feature importance for this cluster
    cluster_profiles <- all_user_profiles[cluster_users, ]
    avg_profile <- colMeans(cluster_profiles)
    
    # Get top features for this cluster
    feature_importance <- data.frame(
      feature = recommender$feature_names,
      importance = abs(avg_profile)
    )
    feature_importance <- feature_importance[order(feature_importance$importance, decreasing = TRUE), ]
    
    cat(sprintf("Top 5 features for Cluster %d:\n", cluster_id))
    for (i in 1:5) {
      cat(sprintf("  %s: %.3f\n", feature_importance$feature[i], feature_importance$importance[i]))
    }
  }
  
  return(list(pca_plot = p1, cluster_plot = p2))
}

# Demonstrate advanced features
demonstrate_advanced_features <- function() {
  cat("=== Advanced Content-Based Features ===\n\n")
  
  # Generate data
  data <- generate_synthetic_movie_data()
  movies_df <- data$movies_df
  ratings_df <- data$ratings_df
  
  # Add more complex features
  movies_df$decade <- (movies_df$year %/% 10) * 10
  movies_df$budget_category <- cut(movies_df$budget, breaks = 3, labels = c("Low", "Medium", "High"))
  movies_df$rating_category <- cut(movies_df$rating, breaks = 3, labels = c("Poor", "Average", "Good"))
  
  # Create recommender with advanced features
  recommender <- content_based_recommender("cosine")
  
  # Use more comprehensive feature set
  feature_columns <- c("genre", "year", "rating", "budget", "director", "decade", "budget_category", "rating_category")
  text_columns <- c("description")
  
  recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
  recommender <- create_user_profiles(recommender, ratings_df, movies_df)
  
  cat("Advanced feature set:\n")
  cat("Number of features:", length(recommender$feature_names), "\n")
  cat("Feature names:", paste(recommender$feature_names[1:10], collapse = ", "), "...\n")
  
  # Generate recommendations
  test_user <- 1
  recommendations <- recommend(recommender, test_user, 10)
  
  cat("\nTop 10 recommendations with advanced features:\n")
  for (i in 1:length(recommendations)) {
    item_idx <- recommendations[[i]]$item_index
    similarity <- recommendations[[i]]$similarity
    movie <- movies_df[item_idx, ]
    cat(sprintf("%d. %s (%s, %d, %s) - Similarity: %.3f\n", 
                i, movie$title, movie$genre, movie$year, movie$budget_category, similarity))
  }
  
  # Analyze feature diversity
  cat("\nFeature Analysis:\n")
  cat("Total features:", length(recommender$feature_names), "\n")
  
  # Count feature types
  categorical_features <- sum(grepl("_", recommender$feature_names) & !grepl("^description_", recommender$feature_names))
  numerical_features <- sum(!grepl("_", recommender$feature_names) & !grepl("^description_", recommender$feature_names))
  text_features <- sum(grepl("^description_", recommender$feature_names))
  
  cat("Categorical features:", categorical_features, "\n")
  cat("Numerical features:", numerical_features, "\n")
  cat("Text features:", text_features, "\n")
  
  return(list(movies_df = movies_df, ratings_df = ratings_df, recommender = recommender))
}

# Demonstrate evaluation metrics
demonstrate_evaluation_metrics <- function() {
  cat("=== Evaluation Metrics ===\n\n")
  
  # Generate data
  data <- generate_synthetic_movie_data()
  movies_df <- data$movies_df
  ratings_df <- data$ratings_df
  
  # Split data for evaluation
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_ratings <- ratings_df[train_indices, ]
  test_ratings <- ratings_df[-train_indices, ]
  
  # Train recommender on training data
  recommender <- content_based_recommender("cosine")
  
  feature_columns <- c("genre", "year", "rating", "budget", "director")
  text_columns <- c("description")
  
  recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
  recommender <- create_user_profiles(recommender, train_ratings, movies_df)
  
  # Evaluate on test data
  precision_scores <- c()
  recall_scores <- c()
  
  test_users <- unique(test_ratings$user_id)[1:10]  # Test on first 10 users
  
  for (user_id in test_users) {
    # Get recommendations
    recommendations <- recommend(recommender, user_id, 10)
    recommended_items <- sapply(recommendations, function(x) x$item_index)
    
    # Get ground truth (items rated 4+ in test set)
    user_test_ratings <- test_ratings[test_ratings$user_id == user_id, ]
    true_items <- user_test_ratings[user_test_ratings$rating >= 4, "movie_id"]
    
    # Convert to item indices
    true_indices <- sapply(true_items, function(id) which(movies_df$movie_id == id))
    
    # Compute precision and recall
    if (length(recommended_items) > 0) {
      precision <- length(intersect(recommended_items, true_indices)) / length(recommended_items)
      precision_scores <- c(precision_scores, precision)
    }
    
    if (length(true_indices) > 0) {
      recall <- length(intersect(recommended_items, true_indices)) / length(true_indices)
      recall_scores <- c(recall_scores, recall)
    }
  }
  
  # Calculate average metrics
  avg_precision <- mean(precision_scores, na.rm = TRUE)
  avg_recall <- mean(recall_scores, na.rm = TRUE)
  f1_score <- 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
  
  cat("Evaluation Results:\n")
  cat("Average Precision@10:", round(avg_precision, 3), "\n")
  cat("Average Recall@10:", round(avg_recall, 3), "\n")
  cat("F1 Score:", round(f1_score, 3), "\n")
  
  # Visualize results
  p1 <- ggplot(data.frame(precision = precision_scores), aes(x = precision)) +
    geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7, color = "black") +
    geom_vline(xintercept = avg_precision, color = "red", linetype = "dashed") +
    labs(title = "Precision Distribution",
         x = "Precision@10", y = "Frequency") +
    theme_minimal()
  
  p2 <- ggplot(data.frame(recall = recall_scores), aes(x = recall)) +
    geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7, color = "black") +
    geom_vline(xintercept = avg_recall, color = "red", linetype = "dashed") +
    labs(title = "Recall Distribution",
         x = "Recall@10", y = "Frequency") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(
    precision = avg_precision,
    recall = avg_recall,
    f1_score = f1_score,
    precision_scores = precision_scores,
    recall_scores = recall_scores
  ))
}

# Demonstrate cold start handling
demonstrate_cold_start <- function() {
  cat("=== Cold Start Handling ===\n\n")
  
  # Generate data
  data <- generate_synthetic_movie_data()
  movies_df <- data$movies_df
  ratings_df <- data$ratings_df
  
  # Create recommender
  recommender <- content_based_recommender("cosine")
  
  feature_columns <- c("genre", "year", "rating", "budget", "director")
  text_columns <- c("description")
  
  recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
  recommender <- create_user_profiles(recommender, ratings_df, movies_df)
  
  # Simulate new user (no ratings)
  new_user_id <- 999
  
  # Strategy 1: Use popular items as initial profile
  popular_items <- ratings_df %>%
    group_by(movie_id) %>%
    summarise(avg_rating = mean(rating)) %>%
    arrange(desc(avg_rating)) %>%
    head(10)
  
  popular_indices <- sapply(popular_items$movie_id, function(id) which(movies_df$movie_id == id))
  popular_profiles <- recommender$item_profiles[popular_indices, ]
  new_user_profile <- colMeans(popular_profiles)
  
  # Add to recommender
  recommender$user_profiles[[as.character(new_user_id)]] <- new_user_profile
  
  # Generate recommendations for new user
  recommendations <- recommend(recommender, new_user_id, 10)
  
  cat("Recommendations for new user (popular items strategy):\n")
  for (i in 1:length(recommendations)) {
    item_idx <- recommendations[[i]]$item_index
    similarity <- recommendations[[i]]$similarity
    movie <- movies_df[item_idx, ]
    cat(sprintf("%d. %s (%s, %d) - Similarity: %.3f\n", 
                i, movie$title, movie$genre, movie$year, similarity))
  }
  
  # Strategy 2: Use genre-based profile
  genre_preferences <- c(Action = 0.8, Drama = 0.3, Comedy = 0.6, Thriller = 0.9, Romance = 0.2)
  
  # Create profile based on genre preferences
  genre_profile <- rep(0, length(recommender$feature_names))
  for (i in 1:length(recommender$feature_names)) {
    feature <- recommender$feature_names[i]
    if (grepl("^genre_", feature)) {
      genre <- gsub("^genre_", "", feature)
      if (genre %in% names(genre_preferences)) {
        genre_profile[i] <- genre_preferences[genre]
      }
    }
  }
  
  # Normalize profile
  genre_profile <- genre_profile / sqrt(sum(genre_profile^2))
  
  # Add to recommender
  recommender$user_profiles[[as.character(new_user_id)]] <- genre_profile
  
  # Generate recommendations
  recommendations <- recommend(recommender, new_user_id, 10)
  
  cat("\nRecommendations for new user (genre preferences strategy):\n")
  for (i in 1:length(recommendations)) {
    item_idx <- recommendations[[i]]$item_index
    similarity <- recommendations[[i]]$similarity
    movie <- movies_df[item_idx, ]
    cat(sprintf("%d. %s (%s, %d) - Similarity: %.3f\n", 
                i, movie$title, movie$genre, movie$year, similarity))
  }
  
  return(recommender)
}

# Demonstrate scalability
demonstrate_scalability <- function() {
  cat("=== Scalability Analysis ===\n\n")
  
  # Test with different dataset sizes
  dataset_sizes <- c(50, 100, 200, 500)
  training_times <- c()
  recommendation_times <- c()
  
  for (size in dataset_sizes) {
    cat("Testing with", size, "movies...\n")
    
    # Generate data
    data <- generate_synthetic_movie_data(n_movies = size, n_users = size %/% 2)
    movies_df <- data$movies_df
    ratings_df <- data$ratings_df
    
    # Time training
    start_time <- Sys.time()
    
    recommender <- content_based_recommender("cosine")
    feature_columns <- c("genre", "year", "rating", "budget", "director")
    text_columns <- c("description")
    
    recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
    recommender <- create_user_profiles(recommender, ratings_df, movies_df)
    
    training_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    training_times <- c(training_times, training_time)
    
    # Time recommendations
    start_time <- Sys.time()
    test_users <- names(recommender$user_profiles)[1:5]
    for (user_id in test_users) {
      recommend(recommender, user_id, 10)
    }
    
    recommendation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    recommendation_times <- c(recommendation_times, recommendation_time)
    
    cat("  Training time:", round(training_time, 3), "s\n")
    cat("  Recommendation time (5 users):", round(recommendation_time, 3), "s\n")
  }
  
  # Visualize scalability
  p1 <- ggplot(data.frame(size = dataset_sizes, time = training_times), aes(x = size, y = time)) +
    geom_line() +
    geom_point() +
    labs(title = "Training Time vs Dataset Size",
         x = "Dataset Size (movies)", y = "Time (seconds)") +
    theme_minimal()
  
  p2 <- ggplot(data.frame(size = dataset_sizes, time = recommendation_times), aes(x = size, y = time)) +
    geom_line() +
    geom_point() +
    labs(title = "Recommendation Time vs Dataset Size",
         x = "Dataset Size (movies)", y = "Time (seconds)") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(
    dataset_sizes = dataset_sizes,
    training_times = training_times,
    recommendation_times = recommendation_times
  ))
}

# Main demonstration function
main_r <- function() {
  cat("Content-Based Recommender System: Implementation and Analysis\n")
  cat("=" * 70, "\n")
  
  # 1. Basic content-based demonstration
  cat("\n1. Basic Content-Based Recommender System:\n")
  basic_data <- demonstrate_basic_content_based()
  
  # 2. Feature importance analysis
  cat("\n2. Feature Importance Analysis:\n")
  feature_plots <- demonstrate_feature_importance()
  
  # 3. Similarity metrics comparison
  cat("\n3. Similarity Metrics Comparison:\n")
  similarity_results <- demonstrate_similarity_metrics()
  
  # 4. Profile visualization
  cat("\n4. Profile Visualization:\n")
  profile_plots <- demonstrate_profile_visualization()
  
  # 5. Advanced features
  cat("\n5. Advanced Features:\n")
  advanced_data <- demonstrate_advanced_features()
  
  # 6. Evaluation metrics
  cat("\n6. Evaluation Metrics:\n")
  evaluation_results <- demonstrate_evaluation_metrics()
  
  # 7. Cold start handling
  cat("\n7. Cold Start Handling:\n")
  cold_start_recommender <- demonstrate_cold_start()
  
  # 8. Scalability analysis
  cat("\n8. Scalability Analysis:\n")
  scalability_results <- demonstrate_scalability()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Content-based filtering leverages item features for recommendations\n")
  cat("2. Feature engineering is crucial for system performance\n")
  cat("3. Different similarity metrics can produce different results\n")
  cat("4. User profiles can be clustered to understand user segments\n")
  cat("5. Advanced features improve recommendation quality\n")
  cat("6. Evaluation requires multiple metrics for comprehensive assessment\n")
  cat("7. Cold start can be handled with various strategies\n")
  cat("8. Scalability becomes important with large datasets\n")
  
  return(list(
    basic_data = basic_data,
    feature_plots = feature_plots,
    similarity_results = similarity_results,
    profile_plots = profile_plots,
    advanced_data = advanced_data,
    evaluation_results = evaluation_results,
    scalability_results = scalability_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
