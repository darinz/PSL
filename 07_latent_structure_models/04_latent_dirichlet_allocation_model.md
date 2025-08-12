# 7.4. Latent Dirichlet Allocation Model

## 7.4.1. Introduction to LDA

Latent Dirichlet Allocation (LDA) is a powerful probabilistic model for discovering the underlying topics in a collection of documents. It was introduced by Blei, Ng, and Jordan in 2003 and has become one of the most widely used topic modeling techniques in natural language processing and text mining.

**Intuitive Understanding**: LDA is like being a librarian who wants to understand the themes and topics in a large collection of books without having to read every single one. Imagine you have a huge library with thousands of books, and you want to figure out what the main themes are. Instead of reading each book cover to cover, you look at the words that appear most frequently in each book and try to group books that use similar vocabulary. LDA is like an intelligent librarian who can automatically discover that some books are about "cooking" (they contain words like "recipe," "ingredients," "kitchen"), others are about "travel" (words like "destination," "hotel," "vacation"), and others are about "technology" (words like "computer," "software," "algorithm"). The librarian doesn't just group books by individual words, but by patterns of words that tend to appear together, revealing the underlying themes or "topics" in the library.

![Graphical model of Latent Dirichlet Allocation (LDA).](../_images/w7_LDA.png)
*Figure: Graphical model of Latent Dirichlet Allocation (LDA).*

### Problem Setup

Consider a collection of $`n`$ documents, where each document is represented as a "bag of words" (ignoring word order). Let $`V`$ be the vocabulary size (number of unique words), then:

- **Document representation**: Each document $`d_i`$ is represented as a vector $`\mathbf{w}_i = (w_{i1}, w_{i2}, \ldots, w_{iV})`$ where $`w_{ij}`$ is the frequency of word $`j`$ in document $`i`$ - like counting how many times each word appears in each book
- **Document collection**: The entire collection is represented as a $`V \times n`$ matrix $`\mathbf{W}`$ - like a giant spreadsheet where each row is a word, each column is a book, and each cell shows how many times that word appears in that book

**Intuition**: This setup is like having a massive library catalog where you've counted every word in every book. Each book becomes a list of word counts, and the entire library becomes a huge table showing how many times each word appears in each book. This gives us a way to represent books mathematically, but it's still very high-dimensional and doesn't capture the underlying themes.

### The Challenge

Direct representation using word frequencies has several limitations:
1. **High dimensionality**: $`V`$ can be very large (thousands to millions of words) - like having a vocabulary so huge that it's overwhelming
2. **Sparsity**: Most documents contain only a small subset of the vocabulary - like most books only using a tiny fraction of all possible words
3. **No semantic structure**: Word frequencies don't capture underlying themes or topics - like knowing that "recipe" appears 10 times and "algorithm" appears 5 times doesn't tell us if the book is about cooking or computer science

**Intuition**: These challenges are like trying to understand a library by just looking at word counts. It's like having a massive dictionary and counting how many times each word appears in each book, but this doesn't help us understand what the books are actually about. We need a way to discover the underlying themes or "topics" that organize the vocabulary.

### LDA Solution

LDA addresses these issues by introducing the concept of **topics**:
- Each topic $`k`$ is a distribution over words: $`\boldsymbol{\beta}_k = (\beta_{k1}, \beta_{k2}, \ldots, \beta_{kV})`$ where $`\beta_{kv}`$ is the probability of word $`v`$ appearing in topic $`k`$ - like defining what words are most likely to appear in a "cooking" topic
- Each document $`d_i`$ has a distribution over topics: $`\boldsymbol{\theta}_i = (\theta_{i1}, \theta_{i2}, \ldots, \theta_{iK})`$ where $`\theta_{ik}`$ is the probability of topic $`k`$ in document $`i`$ - like saying a book is 70% about cooking, 20% about travel, and 10% about technology

**Intuition**: LDA is like discovering that there are underlying "themes" or "topics" in the library. Each topic is like a "word cloud" that shows which words are most likely to appear together (like a cooking topic having "recipe," "ingredients," "kitchen," "chef," etc.). Each book is then described as a mix of these topics - maybe 70% cooking, 20% travel, and 10% technology. This gives us a much more meaningful and compact representation of the library.

## 7.4.2. Mathematical Foundation

### Generative Process

![Plate diagram of the LDA generative process (Blei et al., 2003).](../_images/w7_Blei_2012.png)

*Figure: Plate diagram of the LDA generative process (Blei et al., 2003).* 

LDA assumes the following generative process for creating documents:

1. **For each topic $`k = 1, 2, \ldots, K`$**:
   - Draw a word distribution: $`\boldsymbol{\beta}_k \sim \text{Dirichlet}(\boldsymbol{\eta})`$ - like defining what words are most likely to appear in each theme

2. **For each document $`i = 1, 2, \ldots, n`$**:
   - Draw a topic distribution: $`\boldsymbol{\theta}_i \sim \text{Dirichlet}(\boldsymbol{\alpha})`$ - like deciding what mix of themes this book will have
   - **For each word position $`j = 1, 2, \ldots, N_i`$ in document $`i`$**:
     - Draw a topic assignment: $`z_{ij} \sim \text{Multinomial}(\boldsymbol{\theta}_i)`$ - like choosing which theme this word comes from
     - Draw a word: $`w_{ij} \sim \text{Multinomial}(\boldsymbol{\beta}_{z_{ij}})`$ - like choosing a word from that theme's vocabulary

**Intuition**: This generative process is like imagining how books are created in the library. First, we define what each theme looks like (what words are most likely in cooking, travel, technology, etc.). Then, for each book, we decide what mix of themes it will have (maybe 70% cooking, 20% travel, 10% technology). Finally, for each word in the book, we first choose which theme it comes from (maybe the 5th word comes from the cooking theme), then choose a specific word from that theme's vocabulary (maybe "recipe" from the cooking theme). This process creates books that are coherent mixtures of themes.

### Mathematical Formulation

The joint distribution of all variables is:

$$ p(\mathbf{W}, \mathbf{Z}, \boldsymbol{\Theta}, \mathbf{B} \mid \boldsymbol{\alpha}, \boldsymbol{\eta}) = \prod_{k=1}^K p(\boldsymbol{\beta}_k \mid \boldsymbol{\eta}) \prod_{i=1}^n p(\boldsymbol{\theta}_i \mid \boldsymbol{\alpha}) \prod_{j=1}^{N_i} p(z_{ij} \mid \boldsymbol{\theta}_i) p(w_{ij} \mid \boldsymbol{\beta}_{z_{ij}}) $$

where:
- $`\mathbf{W}`$: observed words - like all the words we see in all the books
- $`\mathbf{Z}`$: topic assignments - like which theme each word came from
- $`\boldsymbol{\Theta}`$: document-topic distributions - like how much each book is about each theme
- $`\mathbf{B}`$: topic-word distributions - like what words are most likely in each theme
- $`\boldsymbol{\alpha}`$: prior for document-topic distributions - like our initial beliefs about how books mix themes
- $`\boldsymbol{\eta}`$: prior for topic-word distributions - like our initial beliefs about what words belong in each theme

**Intuition**: This formula describes the complete probability of everything we observe (the words) and everything we don't observe (which theme each word came from, how much each book is about each theme, what words belong to each theme). It's like describing the complete probability of the entire library, including all the hidden structure that explains why certain words appear together.

### Marginal Likelihood

The marginal likelihood of the observed words is:

$$ p(\mathbf{W} \mid \boldsymbol{\alpha}, \boldsymbol{\eta}) = \int \int \sum_{\mathbf{Z}} p(\mathbf{W}, \mathbf{Z}, \boldsymbol{\Theta}, \mathbf{B} \mid \boldsymbol{\alpha}, \boldsymbol{\eta}) \, d\boldsymbol{\Theta} \, d\mathbf{B} $$

This integral is intractable, which is why we need approximate inference methods.

**Intuition**: The marginal likelihood is like asking "how likely are we to see these words in these books, given our model?" We have to sum over all possible ways the words could have been assigned to themes, and integrate over all possible theme definitions and book-theme mixtures. This is computationally impossible to do exactly, which is why we need clever approximation methods.

## 7.4.3. Implementation: Basic LDA

The basic LDA implementation demonstrates the core concepts of topic modeling with a custom implementation and comparison with scikit-learn. The `LDAModel` class provides a complete implementation including parameter initialization, variational inference, and result extraction.

**Key Functions:**
- `LDAModel.__init__()`: Initialize LDA model with specified parameters - like setting up the librarian's analysis tools
- `LDAModel.fit()`: Fit the model to documents using variational inference - like the librarian analyzing the books to discover themes
- `LDAModel.get_top_words()`: Extract top words for each topic - like the librarian listing the most important words for each theme
- `LDAModel.get_document_topics()`: Get topic distribution for documents - like the librarian describing what mix of themes each book has
- `demonstrate_basic_lda()`: Complete demonstration with synthetic data - like the librarian testing their analysis on a small sample of books

The implementation includes comparison with scikit-learn's `LatentDirichletAllocation` to validate results and demonstrate the relationship between custom and library implementations.

See the implementation in `code/lda_implementation.py` for the complete basic LDA workflow.

The R implementation provides equivalent functionality using the `topicmodels` package, which offers both Gibbs sampling and variational inference methods for LDA. The implementation demonstrates document preprocessing, model fitting, and result extraction.

**Key Functions:**
- `demonstrate_basic_lda()`: Complete R implementation with synthetic data - like the R librarian's analysis tools
- Uses `topicmodels::LDA()` with Gibbs sampling method - like using proven R tools for theme discovery
- Includes topic visualization with wordclouds - like creating visual summaries of each theme
- Extracts both topic-word and document-topic distributions - like getting both theme definitions and book-theme mixtures

The R implementation leverages the `tm` package for text preprocessing and `topicmodels` for the core LDA algorithm, providing a robust and efficient solution for topic modeling in R.

See the implementation in `code/r_lda_implementation.R` for the complete R-based LDA workflow.

## 7.4.4. Variational Inference for LDA

### The Variational Approximation

Since exact inference is intractable, we use variational inference. We approximate the posterior with a factorized distribution:

$$ q(\mathbf{Z}, \boldsymbol{\Theta}, \mathbf{B}) = \prod_{i=1}^n q(\boldsymbol{\theta}_i) \prod_{k=1}^K q(\boldsymbol{\beta}_k) \prod_{i,j} q(z_{ij}) $$

where:
- $`q(\boldsymbol{\theta}_i) = \text{Dirichlet}(\boldsymbol{\gamma}_i)`$ - like approximating each book's theme mixture
- $`q(\boldsymbol{\beta}_k) = \text{Dirichlet}(\boldsymbol{\lambda}_k)`$ - like approximating each theme's word distribution
- $`q(z_{ij}) = \text{Multinomial}(\boldsymbol{\phi}_{ij})`$ - like approximating which theme each word came from

**Intuition**: Variational inference is like the librarian making educated guesses about the hidden structure. Instead of trying to find the exact truth (which is computationally impossible), the librarian makes reasonable approximations: "I think this book is mostly about cooking with some travel, I think the cooking theme includes these words, and I think this word probably came from the cooking theme." These approximations are much easier to work with and still give us good results.

### Variational Updates

The variational parameters are updated iteratively:

**Document-topic distributions**:
$$ \gamma_{ik} = \alpha_k + \sum_{j=1}^{N_i} \phi_{ijk} $$

**Topic-word distributions**:
$$ \lambda_{kv} = \eta_v + \sum_{i=1}^n \sum_{j=1}^{N_i} \phi_{ijk} \mathbf{1}[w_{ij} = v] $$

**Topic assignments**:
$$ \phi_{ijk} \propto \exp\left(\mathbb{E}_{q}[\log \theta_{ik}] + \mathbb{E}_{q}[\log \beta_{k,w_{ij}}]\right) $$

**Intuition**: These updates are like the librarian refining their understanding step by step. The first update says "given my current guesses about which words came from which themes, what do I think about each book's theme mixture?" The second update says "given my current guesses about word-theme assignments, what do I think each theme's word distribution looks like?" The third update says "given my current understanding of themes and book mixtures, which theme do I think each word most likely came from?" Each update improves the librarian's understanding.

### Implementation: Variational LDA

The variational inference implementation provides a more sophisticated approach to LDA parameter estimation using the EM algorithm with variational approximations. The `VariationalLDA` class implements the complete variational inference framework.

**Key Functions:**
- `VariationalLDA.__init__()`: Initialize variational LDA with convergence parameters - like setting up a sophisticated librarian's analysis system
- `VariationalLDA.fit()`: Fit model using variational inference - like the librarian performing advanced theme analysis
- `VariationalLDA._update_phi()`: Update topic assignment distributions - like refining guesses about which theme each word came from
- `VariationalLDA._update_gamma()`: Update document-topic distributions - like refining understanding of each book's theme mixture
- `VariationalLDA._update_lambda()`: Update topic-word distributions - like refining understanding of what words belong to each theme
- `demonstrate_variational_lda()`: Complete demonstration with convergence monitoring - like the librarian testing their advanced analysis system

The variational approach uses digamma functions and log-sum-exp tricks for numerical stability, providing more robust parameter estimation compared to basic implementations.

See the implementation in `code/lda_implementation.py` for the complete variational inference workflow.

## 7.4.5. Gibbs Sampling for LDA

### Gibbs Sampling Algorithm

Gibbs sampling is another popular inference method for LDA. It samples from the posterior distribution by iteratively updating each latent variable conditioned on the others.

**Intuition**: Gibbs sampling is like the librarian taking a different approach to understanding the library. Instead of making educated guesses and refining them, the librarian randomly samples possible explanations and gradually improves them. It's like saying "let me randomly assign each word to a theme, then look at the patterns and adjust my assignments, then look at the patterns again and adjust some more." Over many iterations, this random sampling approach converges to a good understanding of the themes.

### Mathematical Formulation

The conditional distribution for topic assignment $`z_{ij}`$ is:

$$ p(z_{ij} = k \mid \mathbf{z}_{-ij}, \mathbf{w}, \boldsymbol{\alpha}, \boldsymbol{\beta}) \propto \frac{n_{ik}^{-ij} + \alpha_k}{\sum_{k'} (n_{ik'}^{-ij} + \alpha_{k'})} \cdot \frac{n_{kv}^{-ij} + \beta_v}{\sum_{v'} (n_{kv'}^{-ij} + \beta_{v'})} $$

where:
- $`n_{ik}^{-ij}`$: number of words in document $`i`$ assigned to topic $`k`$ (excluding word $`j`$) - like how many other words in this book are currently assigned to theme k
- $`n_{kv}^{-ij}`$: number of times word $`v`$ is assigned to topic $`k`$ (excluding word $`j`$) - like how many times this word appears in theme k across all books

**Intuition**: This formula tells us how likely it is that a particular word came from a particular theme. It considers two things: (1) how much this book is already about this theme (if the book is mostly about cooking, a new word is more likely to be about cooking), and (2) how common this word is in this theme (if "recipe" appears often in the cooking theme, it's more likely that this "recipe" came from cooking). The formula balances these two pieces of evidence to make a good guess.

The Gibbs sampling implementation provides an alternative inference method for LDA using Markov Chain Monte Carlo techniques. The `GibbsSamplingLDA` class implements the complete Gibbs sampling algorithm with topic assignment sampling and parameter estimation.

**Key Functions:**
- `GibbsSamplingLDA.__init__()`: Initialize Gibbs sampling LDA with parameters - like setting up the librarian's sampling-based analysis system
- `GibbsSamplingLDA.fit()`: Fit model using Gibbs sampling - like the librarian performing sampling-based theme analysis
- `GibbsSamplingLDA._initialize_topic_assignments()`: Initialize topic assignments randomly - like the librarian making random initial guesses about word-theme assignments
- `GibbsSamplingLDA._gibbs_sampling()`: Perform Gibbs sampling iterations - like the librarian repeatedly updating their understanding through sampling
- `GibbsSamplingLDA._sample_topic_assignments()`: Sample topic assignments for all words - like the librarian updating all word-theme assignments
- `GibbsSamplingLDA._sample_topic_assignment()`: Sample topic assignment for a single word - like the librarian deciding which theme a single word most likely came from
- `GibbsSamplingLDA._compute_final_distributions()`: Compute final topic-word and document-topic distributions - like the librarian summarizing their final understanding of themes and book mixtures
- `demonstrate_gibbs_sampling_lda()`: Complete demonstration with result analysis - like the librarian testing their sampling-based analysis system

The Gibbs sampling approach provides an alternative to variational inference, often yielding more accurate posterior estimates at the cost of increased computational complexity.

See the implementation in `code/lda_implementation.py` for the complete Gibbs sampling LDA workflow.

## 7.4.6. Model Evaluation and Selection

### Perplexity

Perplexity measures how well the model predicts held-out documents:

$$ \text{Perplexity} = \exp\left(-\frac{\sum_{d=1}^D \log p(\mathbf{w}_d \mid \boldsymbol{\alpha}, \boldsymbol{\beta})}{\sum_{d=1}^D N_d}\right) $$

**Intuition**: Perplexity is like measuring how surprised the librarian is when they see new books. If the librarian has discovered good themes, they should be able to predict what words will appear in new books. Low perplexity means the librarian is not very surprised (good predictions), while high perplexity means the librarian is very surprised (poor predictions). It's like testing whether the librarian's understanding of themes helps them understand new books.

### Coherence Score

Topic coherence measures the semantic similarity of words within a topic:

$$ \text{Coherence} = \sum_{i=2}^M \sum_{j=1}^{i-1} \log \frac{p(w_i, w_j) + \epsilon}{p(w_j)} $$

**Intuition**: Coherence is like measuring how well the words in a theme "go together." A good cooking theme should have words that are semantically related (like "recipe," "ingredients," "kitchen," "chef"). A bad theme might have unrelated words (like "recipe," "computer," "elephant," "mountain"). The coherence score measures how much the words in each theme are semantically related, helping us identify which themes are meaningful and which are just random word collections.

The model evaluation implementation provides comprehensive tools for assessing LDA model quality using perplexity and coherence metrics. The evaluation framework supports systematic comparison of models with different numbers of topics.

**Key Functions:**
- `evaluate_lda_models()`: Evaluate LDA models with different numbers of topics - like the librarian testing different numbers of themes to see which works best
- `compute_topic_coherence()`: Compute topic coherence score for semantic quality assessment - like measuring how well the words in each theme go together
- `compute_word_similarity()`: Compute similarity between words (placeholder implementation) - like measuring how related different words are
- `demonstrate_model_evaluation()`: Complete demonstration with visualization - like the librarian systematically testing and comparing different theme analysis approaches

The evaluation includes both perplexity (predictive performance) and coherence (semantic quality) metrics, providing a comprehensive assessment of topic model quality. The implementation includes visualization of evaluation results to aid in model selection.

See the implementation in `code/lda_implementation.py` for the complete model evaluation workflow.

## 7.4.7. Applications and Extensions

### Document Classification

LDA can be used for document classification by using topic distributions as features. The `lda_classification()` function demonstrates how to use LDA-derived topic features for supervised learning tasks.

**Intuition**: This is like using the librarian's theme analysis to help categorize new books. Instead of trying to classify books based on individual words (which can be noisy and high-dimensional), we use the theme mixtures that the librarian discovered. A book that's 80% about cooking and 20% about travel is more likely to be a cookbook than a pure travel book. This gives us a more robust and interpretable way to classify documents.

**Key Functions:**
- `lda_classification()`: Use LDA topic distributions as features for document classification - like using theme mixtures to categorize books
- Integrates with scikit-learn's cross-validation framework - like systematically testing the classification approach
- Demonstrates the utility of topic modeling for feature engineering - like showing how theme analysis improves document understanding

The implementation shows how topic modeling can enhance document classification by providing interpretable, low-dimensional representations of documents.

See the implementation in `code/lda_implementation.py` for the complete document classification workflow.

### Topic Evolution Over Time

LDA can be extended to model how topics evolve over time. The `temporal_lda()` function implements a simple approach to temporal topic modeling by grouping documents into time windows and fitting separate LDA models.

**Intuition**: This is like the librarian analyzing how themes in the library change over time. Maybe in the 1950s, there were many books about "traditional cooking," but in the 2000s, there are more books about "fusion cuisine" and "molecular gastronomy." The librarian can track how themes emerge, evolve, and disappear over time, revealing the changing interests and trends in the library's collection.

**Key Functions:**
- `temporal_lda()`: Simple temporal LDA implementation using time window grouping - like the librarian analyzing themes in different time periods
- Groups documents by time windows and fits separate LDA models - like analyzing themes in the 1950s, 1960s, 1970s, etc.
- Returns topic-word distributions for each time window - like showing how each theme's vocabulary changed over time

The implementation demonstrates how topic modeling can be extended to capture temporal dynamics in document collections, enabling analysis of how topics emerge, evolve, and disappear over time.

See the implementation in `code/lda_implementation.py` for the complete temporal LDA workflow.

### Hierarchical LDA

Hierarchical LDA extends LDA to model hierarchical topic structures. The `HierarchicalLDA` class implements a simplified version of hierarchical topic modeling that creates multi-level topic hierarchies.

**Intuition**: This is like the librarian discovering that themes have sub-themes. Maybe there's a broad "cooking" theme, but within that, there are sub-themes like "Italian cooking," "French cooking," "Asian cooking," etc. And within "Italian cooking," there might be sub-sub-themes like "pasta dishes," "pizza," "desserts," etc. This creates a hierarchical organization of themes that captures the natural structure of knowledge.

**Key Functions:**
- `HierarchicalLDA.__init__()`: Initialize hierarchical LDA with level and topic parameters - like setting up the librarian's hierarchical analysis system
- `HierarchicalLDA.fit()`: Fit hierarchical LDA using recursive LDA fitting - like the librarian discovering themes at multiple levels of detail
- Creates topic hierarchies by using topic assignments as "documents" for higher levels - like using broad themes to create sub-themes
- Returns topic-word distributions for each level of the hierarchy - like showing the vocabulary for each level of theme organization

The implementation demonstrates how topic modeling can be extended to capture hierarchical relationships between topics, enabling more nuanced analysis of document structure and topic organization.

See the implementation in `code/lda_implementation.py` for the complete hierarchical LDA workflow.

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of LDA and its variants. The code examples demonstrate both the theoretical concepts and their practical application in topic modeling.

---

## Code Files Summary

The LDA concepts have been implemented in the following code files:

### Python Implementation (`code/lda_implementation.py`)
- **Basic LDA**: `LDAModel` class with custom variational inference implementation - like the librarian's basic theme analysis tools
- **Variational LDA**: `VariationalLDA` class with sophisticated variational inference using digamma functions - like the librarian's advanced analysis system
- **Gibbs Sampling LDA**: `GibbsSamplingLDA` class with complete Gibbs sampling implementation - like the librarian's sampling-based analysis approach
- **Model Evaluation**: Functions for perplexity, coherence, and model comparison - like the librarian's quality assessment tools
- **Applications**: Document classification, temporal LDA, and hierarchical LDA implementations - like the librarian's specialized analysis techniques
- **Demonstration Functions**: Complete workflows for each LDA variant - like the librarian's complete analysis procedures

### R Implementation (`code/r_lda_implementation.R`)
- **Basic LDA**: R implementation using `topicmodels` package with Gibbs sampling - like the R librarian's theme analysis tools
- **Document Preprocessing**: Corpus creation and document-term matrix construction - like preparing the library catalog for analysis
- **Model Fitting**: LDA model training with parameter tuning - like the librarian discovering themes in the collection
- **Result Extraction**: Topic-word and document-topic distribution analysis - like extracting theme definitions and book-theme mixtures
- **Visualization**: Topic visualization using wordclouds - like creating visual summaries of each theme

Both implementations provide comprehensive coverage of LDA concepts with practical examples and demonstrate the relationship between theoretical foundations and practical applications in topic modeling.

---

**Navigation:**
- **Next Topic:** [Hidden Markov Models](05_hidden_markov_models.md) - Sequential data and temporal dependencies
- **Previous Topic:** [The EM Algorithm](03_em_algorithm.md) - Expectation-Maximization for latent variables
