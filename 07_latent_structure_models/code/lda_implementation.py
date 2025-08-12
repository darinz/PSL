"""
Latent Dirichlet Allocation (LDA) Implementation
===============================================

This module provides comprehensive implementations of LDA and its variants,
including basic LDA, variational inference, Gibbs sampling, model evaluation,
and various applications and extensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.special import digamma, polygamma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

class LDAModel:
    """
    Basic LDA model implementation with variational inference.
    """
    
    def __init__(self, n_topics=3, alpha=0.1, beta=0.1, max_iter=100, random_state=42):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit(self, documents, vocabulary=None):
        """
        Fit LDA model to documents
        
        Parameters:
        -----------
        documents : list
            List of document strings
        vocabulary : list, optional
            Pre-defined vocabulary
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Preprocess documents
        if vocabulary is None:
            self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            self.word_doc_matrix = self.vectorizer.fit_transform(documents)
            self.vocabulary = self.vectorizer.get_feature_names_out()
        else:
            self.vocabulary = vocabulary
            self.vectorizer = CountVectorizer(vocabulary=vocabulary)
            self.word_doc_matrix = self.vectorizer.fit_transform(documents)
        
        self.n_docs, self.n_words = self.word_doc_matrix.shape
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Run variational inference
        self._variational_inference()
        
        return self
    
    def _initialize_parameters(self):
        """Initialize topic-word and document-topic distributions"""
        # Topic-word distributions (K x V)
        self.beta = np.random.dirichlet([self.beta] * self.n_words, size=self.n_topics)
        
        # Document-topic distributions (D x K)
        self.theta = np.random.dirichlet([self.alpha] * self.n_topics, size=self.n_docs)
        
        # Topic assignments for each word
        self.z = np.random.randint(0, self.n_topics, size=self.word_doc_matrix.nnz)
        
    def _variational_inference(self):
        """Perform variational inference"""
        for iteration in range(self.max_iter):
            # Update topic assignments
            self._update_topic_assignments()
            
            # Update topic-word distributions
            self._update_topic_word_distributions()
            
            # Update document-topic distributions
            self._update_document_topic_distributions()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}")
    
    def _update_topic_assignments(self):
        """Update topic assignments for each word"""
        # This is a simplified version - in practice, you'd use more sophisticated methods
        for doc_idx in range(self.n_docs):
            doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
            for word_idx, word_count in enumerate(doc_words):
                if word_count > 0:
                    # Compute probability of each topic for this word
                    topic_probs = np.zeros(self.n_topics)
                    for topic_idx in range(self.n_topics):
                        topic_probs[topic_idx] = (self.theta[doc_idx, topic_idx] * 
                                                self.beta[topic_idx, word_idx])
                    
                    # Normalize and sample
                    topic_probs = topic_probs / np.sum(topic_probs)
                    self.z[doc_idx * self.n_words + word_idx] = np.random.choice(
                        self.n_topics, p=topic_probs)
    
    def _update_topic_word_distributions(self):
        """Update topic-word distributions"""
        for topic_idx in range(self.n_topics):
            word_counts = np.zeros(self.n_words)
            for doc_idx in range(self.n_docs):
                doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
                for word_idx, word_count in enumerate(doc_words):
                    if word_count > 0 and self.z[doc_idx * self.n_words + word_idx] == topic_idx:
                        word_counts[word_idx] += word_count
            
            # Add prior and normalize
            word_counts += self.beta
            self.beta[topic_idx] = word_counts / np.sum(word_counts)
    
    def _update_document_topic_distributions(self):
        """Update document-topic distributions"""
        for doc_idx in range(self.n_docs):
            topic_counts = np.zeros(self.n_topics)
            for topic_idx in range(self.n_topics):
                for word_idx in range(self.n_words):
                    if self.word_doc_matrix[doc_idx, word_idx] > 0:
                        if self.z[doc_idx * self.n_words + word_idx] == topic_idx:
                            topic_counts[topic_idx] += self.word_doc_matrix[doc_idx, word_idx]
            
            # Add prior and normalize
            topic_counts += self.alpha
            self.theta[doc_idx] = topic_counts / np.sum(topic_counts)
    
    def get_top_words(self, topic_idx, n_words=10):
        """
        Get top words for a given topic
        
        Parameters:
        -----------
        topic_idx : int
            Topic index
        n_words : int
            Number of top words to return
            
        Returns:
        --------
        top_words : list
            List of (word, probability) tuples
        """
        topic_word_probs = self.beta[topic_idx]
        top_word_indices = np.argsort(topic_word_probs)[-n_words:][::-1]
        return [(self.vocabulary[i], topic_word_probs[i]) for i in top_word_indices]
    
    def get_document_topics(self, doc_idx):
        """
        Get topic distribution for a given document
        
        Parameters:
        -----------
        doc_idx : int
            Document index
            
        Returns:
        --------
        doc_topics : array
            Topic distribution for the document
        """
        return self.theta[doc_idx]

def demonstrate_basic_lda():
    """
    Demonstrate basic LDA implementation with synthetic data.
    """
    np.random.seed(42)
    
    # Create synthetic documents
    documents = [
        "machine learning artificial intelligence data science",
        "machine learning algorithms neural networks deep learning",
        "artificial intelligence robotics automation technology",
        "data science statistics analysis visualization",
        "business finance economics market investment",
        "business strategy management leadership",
        "finance banking stocks bonds investment",
        "technology software programming coding",
        "technology innovation startup entrepreneurship",
        "science research discovery experiment laboratory"
    ]
    
    # Fit LDA model
    lda = LDAModel(n_topics=3, alpha=0.1, beta=0.1, max_iter=50)
    lda.fit(documents)
    
    # Display results
    print("Top words for each topic:")
    for topic_idx in range(lda.n_topics):
        top_words = lda.get_top_words(topic_idx, n_words=5)
        print(f"Topic {topic_idx + 1}: {[word for word, prob in top_words]}")
    
    print("\nDocument-topic distributions:")
    for doc_idx in range(min(5, lda.n_docs)):
        doc_topics = lda.get_document_topics(doc_idx)
        print(f"Document {doc_idx + 1}: {doc_topics}")
    
    # Compare with sklearn implementation
    sklearn_lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=50)
    sklearn_lda.fit(lda.word_doc_matrix)
    
    print("\nSklearn LDA results:")
    feature_names = lda.vocabulary
    for topic_idx, topic in enumerate(sklearn_lda.components_):
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx + 1}: {top_words}")
    
    return lda

class VariationalLDA:
    """
    LDA implementation using variational inference.
    """
    
    def __init__(self, n_topics=3, alpha=0.1, beta=0.1, max_iter=100, tol=1e-6):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, word_doc_matrix, vocabulary):
        """
        Fit LDA using variational inference
        
        Parameters:
        -----------
        word_doc_matrix : sparse matrix
            Document-term matrix
        vocabulary : list
            List of vocabulary words
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.word_doc_matrix = word_doc_matrix
        self.vocabulary = vocabulary
        self.n_docs, self.n_words = word_doc_matrix.shape
        
        # Initialize variational parameters
        self._initialize_variational_parameters()
        
        # Run variational inference
        self._variational_inference()
        
        return self
    
    def _initialize_variational_parameters(self):
        """Initialize variational parameters"""
        # Document-topic distributions: gamma (D x K)
        self.gamma = np.random.gamma(100, 1/100, size=(self.n_docs, self.n_topics))
        
        # Topic-word distributions: lambda (K x V)
        self.lambda_ = np.random.gamma(100, 1/100, size=(self.n_topics, self.n_words))
        
        # Topic assignments: phi (D x N x K) - simplified for sparse representation
        self.phi = {}
        for doc_idx in range(self.n_docs):
            doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
            for word_idx, word_count in enumerate(doc_words):
                if word_count > 0:
                    key = (doc_idx, word_idx)
                    self.phi[key] = np.random.dirichlet([1.0] * self.n_topics)
    
    def _variational_inference(self):
        """Perform variational inference"""
        for iteration in range(self.max_iter):
            old_gamma = self.gamma.copy()
            
            # Update phi (topic assignments)
            self._update_phi()
            
            # Update gamma (document-topic distributions)
            self._update_gamma()
            
            # Update lambda (topic-word distributions)
            self._update_lambda()
            
            # Check convergence
            if np.mean(np.abs(self.gamma - old_gamma)) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}")
    
    def _update_phi(self):
        """Update topic assignment distributions"""
        for doc_idx in range(self.n_docs):
            doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
            for word_idx, word_count in enumerate(doc_words):
                if word_count > 0:
                    key = (doc_idx, word_idx)
                    
                    # Compute unnormalized phi
                    log_phi = np.zeros(self.n_topics)
                    for topic_idx in range(self.n_topics):
                        # E[log theta_ik]
                        log_phi[topic_idx] += digamma(self.gamma[doc_idx, topic_idx])
                        
                        # E[log beta_kv]
                        log_phi[topic_idx] += digamma(self.lambda_[topic_idx, word_idx])
                    
                    # Normalize using log-sum-exp trick
                    max_log = np.max(log_phi)
                    exp_log_phi = np.exp(log_phi - max_log)
                    self.phi[key] = exp_log_phi / np.sum(exp_log_phi)
    
    def _update_gamma(self):
        """Update document-topic distributions"""
        for doc_idx in range(self.n_docs):
            for topic_idx in range(self.n_topics):
                # Prior
                self.gamma[doc_idx, topic_idx] = self.alpha
                
                # Sum over words in document
                doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
                for word_idx, word_count in enumerate(doc_words):
                    if word_count > 0:
                        key = (doc_idx, word_idx)
                        self.gamma[doc_idx, topic_idx] += word_count * self.phi[key][topic_idx]
    
    def _update_lambda(self):
        """Update topic-word distributions"""
        for topic_idx in range(self.n_topics):
            for word_idx in range(self.n_words):
                # Prior
                self.lambda_[topic_idx, word_idx] = self.beta
                
                # Sum over documents
                for doc_idx in range(self.n_docs):
                    doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
                    if doc_words[word_idx] > 0:
                        key = (doc_idx, word_idx)
                        self.lambda_[topic_idx, word_idx] += doc_words[word_idx] * self.phi[key][topic_idx]
    
    def get_topic_word_distributions(self):
        """Get topic-word distributions"""
        topic_word_probs = np.zeros((self.n_topics, self.n_words))
        for k in range(self.n_topics):
            topic_word_probs[k] = self.lambda_[k] / np.sum(self.lambda_[k])
        return topic_word_probs
    
    def get_document_topic_distributions(self):
        """Get document-topic distributions"""
        doc_topic_probs = np.zeros((self.n_docs, self.n_topics))
        for i in range(self.n_docs):
            doc_topic_probs[i] = self.gamma[i] / np.sum(self.gamma[i])
        return doc_topic_probs

def demonstrate_variational_lda(lda_model):
    """
    Demonstrate variational LDA implementation.
    """
    # Fit variational LDA
    vlda = VariationalLDA(n_topics=3, alpha=0.1, beta=0.1, max_iter=100)
    vlda.fit(lda_model.word_doc_matrix, lda_model.vocabulary)
    
    # Get results
    topic_word_probs = vlda.get_topic_word_distributions()
    doc_topic_probs = vlda.get_document_topic_distributions()
    
    print("Variational LDA Results:")
    print("Top words for each topic:")
    for topic_idx in range(vlda.n_topics):
        top_word_indices = np.argsort(topic_word_probs[topic_idx])[-5:][::-1]
        top_words = [vlda.vocabulary[i] for i in top_word_indices]
        print(f"Topic {topic_idx + 1}: {top_words}")
    
    print("\nDocument-topic distributions:")
    for doc_idx in range(min(5, vlda.n_docs)):
        print(f"Document {doc_idx + 1}: {doc_topic_probs[doc_idx]}")
    
    return vlda

class GibbsSamplingLDA:
    """
    LDA implementation using Gibbs sampling.
    """
    
    def __init__(self, n_topics=3, alpha=0.1, beta=0.1, n_iterations=1000, burn_in=100):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        
    def fit(self, word_doc_matrix, vocabulary):
        """
        Fit LDA using Gibbs sampling
        
        Parameters:
        -----------
        word_doc_matrix : sparse matrix
            Document-term matrix
        vocabulary : list
            List of vocabulary words
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.word_doc_matrix = word_doc_matrix
        self.vocabulary = vocabulary
        self.n_docs, self.n_words = word_doc_matrix.shape
        
        # Initialize topic assignments
        self._initialize_topic_assignments()
        
        # Run Gibbs sampling
        self._gibbs_sampling()
        
        return self
    
    def _initialize_topic_assignments(self):
        """Initialize topic assignments randomly"""
        self.z = {}  # topic assignments
        self.n_dk = np.zeros((self.n_docs, self.n_topics))  # document-topic counts
        self.n_kv = np.zeros((self.n_topics, self.n_words))  # topic-word counts
        
        for doc_idx in range(self.n_docs):
            doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
            for word_idx, word_count in enumerate(doc_words):
                if word_count > 0:
                    for _ in range(int(word_count)):
                        topic = np.random.randint(0, self.n_topics)
                        self.z[(doc_idx, word_idx, _)] = topic
                        self.n_dk[doc_idx, topic] += 1
                        self.n_kv[topic, word_idx] += 1
    
    def _gibbs_sampling(self):
        """Perform Gibbs sampling"""
        for iteration in range(self.n_iterations):
            # Sample topic assignments
            self._sample_topic_assignments()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}")
        
        # Compute final distributions
        self._compute_final_distributions()
    
    def _sample_topic_assignments(self):
        """Sample topic assignments for all words"""
        for doc_idx in range(self.n_docs):
            doc_words = self.word_doc_matrix[doc_idx].toarray().flatten()
            for word_idx, word_count in enumerate(doc_words):
                if word_count > 0:
                    for word_instance in range(int(word_count)):
                        key = (doc_idx, word_idx, word_instance)
                        old_topic = self.z[key]
                        
                        # Remove current assignment
                        self.n_dk[doc_idx, old_topic] -= 1
                        self.n_kv[old_topic, word_idx] -= 1
                        
                        # Sample new topic
                        new_topic = self._sample_topic_assignment(doc_idx, word_idx)
                        
                        # Update assignment
                        self.z[key] = new_topic
                        self.n_dk[doc_idx, new_topic] += 1
                        self.n_kv[new_topic, word_idx] += 1
    
    def _sample_topic_assignment(self, doc_idx, word_idx):
        """Sample topic assignment for a single word"""
        # Compute unnormalized probabilities
        probs = np.zeros(self.n_topics)
        for topic_idx in range(self.n_topics):
            # Document-topic term
            probs[topic_idx] = (self.n_dk[doc_idx, topic_idx] + self.alpha) / \
                              (np.sum(self.n_dk[doc_idx, :]) + self.n_topics * self.alpha)
            
            # Topic-word term
            probs[topic_idx] *= (self.n_kv[topic_idx, word_idx] + self.beta) / \
                               (np.sum(self.n_kv[topic_idx, :]) + self.n_words * self.beta)
        
        # Normalize and sample
        probs = probs / np.sum(probs)
        return np.random.choice(self.n_topics, p=probs)
    
    def _compute_final_distributions(self):
        """Compute final topic-word and document-topic distributions"""
        # Topic-word distributions
        self.topic_word_probs = np.zeros((self.n_topics, self.n_words))
        for k in range(self.n_topics):
            self.topic_word_probs[k] = (self.n_kv[k] + self.beta) / \
                                      (np.sum(self.n_kv[k]) + self.n_words * self.beta)
        
        # Document-topic distributions
        self.doc_topic_probs = np.zeros((self.n_docs, self.n_topics))
        for i in range(self.n_docs):
            self.doc_topic_probs[i] = (self.n_dk[i] + self.alpha) / \
                                     (np.sum(self.n_dk[i]) + self.n_topics * self.alpha)
    
    def get_top_words(self, topic_idx, n_words=10):
        """
        Get top words for a given topic
        
        Parameters:
        -----------
        topic_idx : int
            Topic index
        n_words : int
            Number of top words to return
            
        Returns:
        --------
        top_words : list
            List of (word, probability) tuples
        """
        top_word_indices = np.argsort(self.topic_word_probs[topic_idx])[-n_words:][::-1]
        return [(self.vocabulary[i], self.topic_word_probs[topic_idx, i]) 
                for i in top_word_indices]

def demonstrate_gibbs_sampling_lda(lda_model):
    """
    Demonstrate Gibbs sampling LDA implementation.
    """
    # Fit Gibbs sampling LDA
    gibbs_lda = GibbsSamplingLDA(n_topics=3, alpha=0.1, beta=0.1, n_iterations=500, burn_in=100)
    gibbs_lda.fit(lda_model.word_doc_matrix, lda_model.vocabulary)
    
    print("Gibbs Sampling LDA Results:")
    print("Top words for each topic:")
    for topic_idx in range(gibbs_lda.n_topics):
        top_words = gibbs_lda.get_top_words(topic_idx, n_words=5)
        print(f"Topic {topic_idx + 1}: {[word for word, prob in top_words]}")
    
    print("\nDocument-topic distributions:")
    for doc_idx in range(min(5, gibbs_lda.n_docs)):
        print(f"Document {doc_idx + 1}: {gibbs_lda.doc_topic_probs[doc_idx]}")
    
    return gibbs_lda

def evaluate_lda_models(word_doc_matrix, vocabulary, n_topics_range=[2, 3, 4, 5]):
    """
    Evaluate LDA models with different numbers of topics
    
    Parameters:
    -----------
    word_doc_matrix : sparse matrix
        Document-term matrix
    vocabulary : list
        List of vocabulary words
    n_topics_range : list
        Range of topic numbers to evaluate
        
    Returns:
    --------
    results : list
        List of evaluation results
    """
    results = []
    
    # Split data
    train_dtm, test_dtm = train_test_split(word_doc_matrix, test_size=0.2, random_state=42)
    
    for n_topics in n_topics_range:
        print(f"Evaluating model with {n_topics} topics...")
        
        # Fit model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
        lda.fit(train_dtm)
        
        # Compute perplexity
        perplexity = lda.perplexity(test_dtm)
        
        # Compute coherence (simplified)
        coherence = compute_topic_coherence(lda, vocabulary)
        
        results.append({
            'n_topics': n_topics,
            'perplexity': perplexity,
            'coherence': coherence
        })
    
    return results

def compute_topic_coherence(lda_model, vocabulary, n_words=10):
    """
    Compute topic coherence score
    
    Parameters:
    -----------
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    vocabulary : list
        List of vocabulary words
    n_words : int
        Number of top words to consider
        
    Returns:
    --------
    coherence : float
        Average topic coherence score
    """
    coherence_scores = []
    
    for topic_idx in range(lda_model.n_components_):
        # Get top words for topic
        topic_word_probs = lda_model.components_[topic_idx]
        top_word_indices = np.argsort(topic_word_probs)[-n_words:][::-1]
        top_words = [vocabulary[i] for i in top_word_indices]
        
        # Compute pairwise similarities (simplified)
        topic_coherence = 0
        for i in range(1, len(top_words)):
            for j in range(i):
                # Use word co-occurrence as similarity measure
                similarity = compute_word_similarity(top_words[i], top_words[j])
                topic_coherence += similarity
        
        coherence_scores.append(topic_coherence)
    
    return np.mean(coherence_scores)

def compute_word_similarity(word1, word2):
    """
    Compute similarity between two words (simplified)
    
    Parameters:
    -----------
    word1 : str
        First word
    word2 : str
        Second word
        
    Returns:
    --------
    similarity : float
        Similarity score
    """
    # In practice, you'd use word embeddings or co-occurrence statistics
    return 0.1  # Placeholder

def demonstrate_model_evaluation(lda_model):
    """
    Demonstrate LDA model evaluation.
    """
    # Evaluate models
    evaluation_results = evaluate_lda_models(lda_model.word_doc_matrix, lda_model.vocabulary)
    
    print("Model Evaluation Results:")
    for result in evaluation_results:
        print(f"Topics: {result['n_topics']}, "
              f"Perplexity: {result['perplexity']:.2f}, "
              f"Coherence: {result['coherence']:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    n_topics = [r['n_topics'] for r in evaluation_results]
    perplexities = [r['perplexity'] for r in evaluation_results]
    plt.plot(n_topics, perplexities, 'bo-')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs Number of Topics')
    
    plt.subplot(1, 2, 2)
    coherences = [r['coherence'] for r in evaluation_results]
    plt.plot(n_topics, coherences, 'ro-')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.title('Coherence vs Number of Topics')
    
    plt.tight_layout()
    plt.show()
    
    return evaluation_results

def lda_classification(word_doc_matrix, labels, n_topics=3):
    """
    Use LDA for document classification
    
    Parameters:
    -----------
    word_doc_matrix : sparse matrix
        Document-term matrix
    labels : array
        Document labels
    n_topics : int
        Number of topics
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    """
    # Fit LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_features = lda.fit_transform(word_doc_matrix)
    
    # Train classifier
    classifier = LogisticRegression(random_state=42)
    scores = cross_val_score(classifier, topic_features, labels, cv=5)
    
    return scores.mean()

def temporal_lda(documents, timestamps, n_topics=3, time_windows=5):
    """
    Simple temporal LDA implementation
    
    Parameters:
    -----------
    documents : list
        List of document strings
    timestamps : list
        List of timestamps
    n_topics : int
        Number of topics
    time_windows : int
        Number of time windows
        
    Returns:
    --------
    temporal_topics : dict
        Dictionary of topics for each time window
    """
    # Group documents by time windows
    time_groups = {}
    for doc, timestamp in zip(documents, timestamps):
        window = timestamp // time_windows
        if window not in time_groups:
            time_groups[window] = []
        time_groups[window].append(doc)
    
    # Fit LDA for each time window
    temporal_topics = {}
    for window, window_docs in time_groups.items():
        vectorizer = CountVectorizer(max_features=1000)
        dtm = vectorizer.fit_transform(window_docs)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        
        temporal_topics[window] = lda.components_
    
    return temporal_topics

class HierarchicalLDA:
    """
    Hierarchical LDA implementation (simplified).
    """
    
    def __init__(self, n_topics_per_level=3, n_levels=2):
        self.n_topics_per_level = n_topics_per_level
        self.n_levels = n_levels
        
    def fit(self, documents):
        """
        Fit hierarchical LDA (simplified implementation)
        
        Parameters:
        -----------
        documents : list
            List of document strings
            
        Returns:
        --------
        self : object
            Returns self
        """
        # This is a simplified version - full hLDA is more complex
        self.level_topics = []
        
        for level in range(self.n_levels):
            # Fit LDA at this level
            vectorizer = CountVectorizer(max_features=1000)
            dtm = vectorizer.fit_transform(documents)
            
            lda = LatentDirichletAllocation(
                n_components=self.n_topics_per_level, 
                random_state=42
            )
            lda.fit(dtm)
            
            self.level_topics.append(lda.components_)
            
            # Use topic assignments to create "documents" for next level
            topic_assignments = lda.transform(dtm)
            documents = [f"topic_{i}" for i in np.argmax(topic_assignments, axis=1)]
        
        return self

def demonstrate_applications(lda_model):
    """
    Demonstrate LDA applications.
    """
    # Document classification
    labels = np.random.randint(0, 2, size=lda_model.word_doc_matrix.shape[0])
    classification_score = lda_classification(lda_model.word_doc_matrix, labels)
    print(f"Classification accuracy: {classification_score:.3f}")
    
    # Temporal LDA (with synthetic timestamps)
    documents = [
        "machine learning artificial intelligence data science",
        "machine learning algorithms neural networks deep learning",
        "artificial intelligence robotics automation technology",
        "data science statistics analysis visualization",
        "business finance economics market investment",
        "business strategy management leadership",
        "finance banking stocks bonds investment",
        "technology software programming coding",
        "technology innovation startup entrepreneurship",
        "science research discovery experiment laboratory"
    ]
    timestamps = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    
    temporal_topics = temporal_lda(documents, timestamps)
    print(f"Temporal LDA fitted for {len(temporal_topics)} time windows")
    
    # Hierarchical LDA
    hlda = HierarchicalLDA(n_topics_per_level=2, n_levels=2)
    hlda.fit(documents)
    print(f"Hierarchical LDA fitted with {len(hlda.level_topics)} levels")

if __name__ == "__main__":
    print("Demonstrating LDA Implementation...")
    
    # Basic LDA
    print("\n1. Basic LDA Implementation")
    lda_model = demonstrate_basic_lda()
    
    # Variational LDA
    print("\n2. Variational LDA Implementation")
    vlda_model = demonstrate_variational_lda(lda_model)
    
    # Gibbs Sampling LDA
    print("\n3. Gibbs Sampling LDA Implementation")
    gibbs_model = demonstrate_gibbs_sampling_lda(lda_model)
    
    # Model Evaluation
    print("\n4. Model Evaluation")
    evaluation_results = demonstrate_model_evaluation(lda_model)
    
    # Applications
    print("\n5. LDA Applications")
    demonstrate_applications(lda_model)
