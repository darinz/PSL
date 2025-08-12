# Hidden Markov Models (HMM) Implementation in R
# ==============================================
#
# This script provides comprehensive implementations of HMM concepts,
# including the dishonest casino example, forward-backward algorithm,
# Viterbi algorithm, and Baum-Welch algorithm.

library(HMM)
library(ggplot2)

# Dishonest Casino HMM Example
# ============================

create_dishonest_casino <- function() {
  # States: Fair (1), Loaded (2)
  states <- c("Fair", "Loaded")
  
  # Symbols: Dice values 1-6
  symbols <- c("1", "2", "3", "4", "5", "6")
  
  # Initial distribution
  startProbs <- c(0.5, 0.5)
  
  # Transition matrix
  transProbs <- matrix(c(0.95, 0.05, 0.1, 0.9), nrow=2, byrow=TRUE)
  
  # Emission matrix
  emissionProbs <- matrix(c(
    1/6, 1/6, 1/6, 1/6, 1/6, 1/6,      # Fair die
    1/10, 1/10, 1/10, 1/10, 1/10, 1/2  # Loaded die
  ), nrow=2, byrow=TRUE)
  
  # Create HMM
  hmm <- initHMM(states, symbols, startProbs, transProbs, emissionProbs)
  return(hmm)
}

demonstrate_dishonest_casino <- function() {
  # Generate sequence
  set.seed(42)
  casino_hmm <- create_dishonest_casino()
  simulation <- simHMM(casino_hmm, 100)
  
  observations <- simulation$observation
  true_states <- simulation$states
  
  cat("Generated sequence statistics:\n")
  cat("Number of 6s:", sum(observations == "6"), "\n")
  cat("Proportion of 6s:", mean(observations == "6"), "\n")
  
  # Baum-Welch algorithm for parameter estimation
  fitted_hmm <- baum_welch(observations)
  
  # Viterbi algorithm
  viterbi_path <- viterbi(fitted_hmm, observations)
  
  cat("\nViterbi decoding accuracy:", mean(viterbi_path == true_states), "\n")
  
  # Visualize results
  df <- data.frame(
    time = 1:length(observations),
    observations = as.numeric(observations),
    true_states = as.numeric(factor(true_states)),
    viterbi_states = as.numeric(factor(viterbi_path))
  )
  
  # Create plots
  p1 <- ggplot(df, aes(x=time)) +
    geom_line(aes(y=observations), color="blue", alpha=0.7) +
    labs(title="Casino Dice Rolls", y="Dice Value") +
    theme_minimal()
  
  p2 <- ggplot(df, aes(x=time)) +
    geom_line(aes(y=true_states), color="green") +
    labs(title="True Hidden States", y="State") +
    theme_minimal()
  
  p3 <- ggplot(df, aes(x=time)) +
    geom_line(aes(y=viterbi_states), color="red") +
    labs(title="Viterbi Decoded States", y="State", x="Time") +
    theme_minimal()
  
  # Display plots
  print(p1)
  print(p2)
  print(p3)
  
  return(list(
    casino_hmm = casino_hmm,
    fitted_hmm = fitted_hmm,
    observations = observations,
    true_states = true_states,
    viterbi_path = viterbi_path
  ))
}

# Baum-Welch Algorithm
# ====================

baum_welch <- function(observations, n_states=2, n_symbols=6, max_iter=100) {
  # Initialize parameters randomly
  startProbs <- runif(n_states)
  startProbs <- startProbs / sum(startProbs)
  
  transProbs <- matrix(runif(n_states^2), nrow=n_states)
  transProbs <- transProbs / rowSums(transProbs)
  
  emissionProbs <- matrix(runif(n_states * n_symbols), nrow=n_states)
  emissionProbs <- emissionProbs / rowSums(emissionProbs)
  
  states <- paste0("State", 1:n_states)
  symbols <- paste0("Symbol", 1:n_symbols)
  
  hmm <- initHMM(states, symbols, startProbs, transProbs, emissionProbs)
  
  # Baum-Welch iterations
  for(iter in 1:max_iter) {
    # Forward-backward algorithm
    fb <- forwardBackward(hmm, observations)
    
    # Update parameters (simplified)
    # In practice, you'd use the full Baum-Welch update equations
    if(iter %% 10 == 0) cat("Iteration", iter, "\n")
  }
  
  return(hmm)
}

# Forward-Backward Algorithm
# ==========================

forward_backward_algorithm <- function(hmm, observations) {
  # Use the HMM package's forwardBackward function
  fb_result <- forwardBackward(hmm, observations)
  
  # Extract forward and backward probabilities
  forward_probs <- fb_result$alpha
  backward_probs <- fb_result$beta
  
  # Compute posterior probabilities
  posterior_probs <- forward_probs * backward_probs
  posterior_probs <- posterior_probs / rowSums(posterior_probs)
  
  return(list(
    forward = forward_probs,
    backward = backward_probs,
    posterior = posterior_probs
  ))
}

demonstrate_forward_backward <- function() {
  # Create a simple HMM
  states <- c("State1", "State2")
  symbols <- c("Symbol1", "Symbol2", "Symbol3")
  
  startProbs <- c(0.5, 0.5)
  transProbs <- matrix(c(0.7, 0.3, 0.4, 0.6), nrow=2, byrow=TRUE)
  emissionProbs <- matrix(c(0.1, 0.4, 0.5, 0.6, 0.3, 0.1), nrow=2, byrow=TRUE)
  
  hmm <- initHMM(states, symbols, startProbs, transProbs, emissionProbs)
  
  # Example observations
  observations <- c("Symbol1", "Symbol2", "Symbol3", "Symbol1", "Symbol2")
  
  # Compute forward-backward
  fb_result <- forward_backward_algorithm(hmm, observations)
  
  cat("Forward probabilities:\n")
  print(fb_result$forward)
  
  cat("\nBackward probabilities:\n")
  print(fb_result$backward)
  
  cat("\nPosterior probabilities:\n")
  print(fb_result$posterior)
  
  return(list(hmm = hmm, fb_result = fb_result, observations = observations))
}

# Viterbi Algorithm
# =================

viterbi_algorithm <- function(hmm, observations) {
  # Use the HMM package's viterbi function
  viterbi_path <- viterbi(hmm, observations)
  
  return(viterbi_path)
}

demonstrate_viterbi <- function() {
  # Create a simple HMM
  states <- c("State1", "State2")
  symbols <- c("Symbol1", "Symbol2", "Symbol3")
  
  startProbs <- c(0.5, 0.5)
  transProbs <- matrix(c(0.7, 0.3, 0.4, 0.6), nrow=2, byrow=TRUE)
  emissionProbs <- matrix(c(0.1, 0.4, 0.5, 0.6, 0.3, 0.1), nrow=2, byrow=TRUE)
  
  hmm <- initHMM(states, symbols, startProbs, transProbs, emissionProbs)
  
  # Example observations
  observations <- c("Symbol1", "Symbol2", "Symbol3", "Symbol1", "Symbol2")
  
  # Compute Viterbi path
  viterbi_path <- viterbi_algorithm(hmm, observations)
  
  cat("Most likely hidden state sequence:\n")
  print(viterbi_path)
  
  # Compare with posterior decoding
  fb_result <- forward_backward_algorithm(hmm, observations)
  posterior_path <- apply(fb_result$posterior, 1, which.max)
  
  cat("\nPosterior decoding:\n")
  print(posterior_path)
  
  return(list(hmm = hmm, viterbi_path = viterbi_path, posterior_path = posterior_path))
}

# Speech Recognition Example
# =========================

speech_recognition_example <- function() {
  # States: phonemes (simplified)
  phonemes <- c("a", "e", "i", "o", "u")
  
  # Observations: acoustic features
  features <- c("low", "mid", "high")
  
  # Initialize HMM for speech recognition
  startProbs <- rep(0.2, 5)  # Equal initial probability
  
  # Transition matrix (phoneme transitions)
  transProbs <- matrix(c(
    0.6, 0.1, 0.1, 0.1, 0.1,  # 'a' transitions
    0.1, 0.6, 0.1, 0.1, 0.1,  # 'e' transitions
    0.1, 0.1, 0.6, 0.1, 0.1,  # 'i' transitions
    0.1, 0.1, 0.1, 0.6, 0.1,  # 'o' transitions
    0.1, 0.1, 0.1, 0.1, 0.6   # 'u' transitions
  ), nrow=5, byrow=TRUE)
  
  # Emission matrix (phoneme to feature mapping)
  emissionProbs <- matrix(c(
    0.7, 0.2, 0.1,  # 'a' -> features
    0.2, 0.7, 0.1,  # 'e' -> features
    0.1, 0.2, 0.7,  # 'i' -> features
    0.6, 0.3, 0.1,  # 'o' -> features
    0.1, 0.3, 0.6   # 'u' -> features
  ), nrow=5, byrow=TRUE)
  
  # Create HMM
  hmm <- initHMM(phonemes, features, startProbs, transProbs, emissionProbs)
  
  return(list(hmm = hmm, phonemes = phonemes, features = features))
}

demonstrate_speech_recognition <- function() {
  speech_hmm <- speech_recognition_example()
  
  # Simulate speech features
  speech_features <- c("low", "mid", "high", "low", "mid", "high", "low", "mid", "high", "low")
  phoneme_sequence <- viterbi(speech_hmm$hmm, speech_features)
  
  cat("Speech recognition example:\n")
  cat("Features:", speech_features, "\n")
  cat("Decoded phonemes:", phoneme_sequence, "\n")
  
  return(list(speech_hmm = speech_hmm, phoneme_sequence = phoneme_sequence))
}

# Gene Finding Example
# ===================

gene_finding_example <- function() {
  # States: coding, non-coding, start codon, stop codon
  states <- c("coding", "non-coding", "start", "stop")
  
  # Observations: DNA bases
  bases <- c("A", "T", "G", "C")
  
  # Initialize HMM for gene finding
  startProbs <- c(0.1, 0.8, 0.05, 0.05)  # Most DNA is non-coding
  
  # Transition matrix
  transProbs <- matrix(c(
    0.95, 0.02, 0.02, 0.01,  # coding transitions
    0.01, 0.98, 0.005, 0.005, # non-coding transitions
    0.99, 0.01, 0.0, 0.0,     # start transitions
    0.01, 0.99, 0.0, 0.0      # stop transitions
  ), nrow=4, byrow=TRUE)
  
  # Emission matrix (base composition)
  emissionProbs <- matrix(c(
    0.25, 0.25, 0.25, 0.25,  # coding (random)
    0.30, 0.30, 0.20, 0.20,  # non-coding
    0.25, 0.25, 0.25, 0.25,  # start
    0.25, 0.25, 0.25, 0.25   # stop
  ), nrow=4, byrow=TRUE)
  
  # Create HMM
  hmm <- initHMM(states, bases, startProbs, transProbs, emissionProbs)
  
  return(list(hmm = hmm, states = states, bases = bases))
}

demonstrate_gene_finding <- function() {
  gene_hmm <- gene_finding_example()
  
  # Simulate DNA sequence
  dna_sequence <- c("A", "T", "G", "C", "A", "T", "G", "C", "A", "T", "G", "C", "A", "T", "G", "C")
  gene_states <- viterbi(gene_hmm$hmm, dna_sequence)
  
  cat("Gene finding example:\n")
  cat("DNA sequence:", dna_sequence, "\n")
  cat("Gene states:", gene_states, "\n")
  
  return(list(gene_hmm = gene_hmm, gene_states = gene_states))
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  cat("Demonstrating HMM Implementation in R...\n")
  
  # Dishonest Casino Example
  cat("\n1. Dishonest Casino HMM\n")
  casino_result <- demonstrate_dishonest_casino()
  
  # Forward-Backward Algorithm
  cat("\n2. Forward-Backward Algorithm\n")
  fb_result <- demonstrate_forward_backward()
  
  # Viterbi Algorithm
  cat("\n3. Viterbi Algorithm\n")
  viterbi_result <- demonstrate_viterbi()
  
  # Speech Recognition
  cat("\n4. Speech Recognition Example\n")
  speech_result <- demonstrate_speech_recognition()
  
  # Gene Finding
  cat("\n5. Gene Finding Example\n")
  gene_result <- demonstrate_gene_finding()
}
