# 12.1. Introduction

Classification trees are a fundamental machine learning technique that extends the concept of decision trees from regression to classification problems. Just as in our previous discussion about regression trees, when it comes to classification trees, we must also focus on three essential aspects:

**Intuitive Understanding**: Classification trees are like building a smart decision-making system that asks a series of yes/no questions to categorize things. Imagine you're trying to identify different types of animals at a zoo. You might start by asking "Is it bigger than a cat?" If yes, you ask "Does it have a trunk?" If yes, it's an elephant. If no, you ask "Does it have stripes?" And so on. Each question splits the animals into two groups, and you keep asking questions until you can confidently identify the animal. Classification trees work the same way - they learn the best questions to ask and the best order to ask them to accurately classify new examples.

### Why Classification Trees Matter

**Intuition**: Classification trees are particularly powerful because they mimic how humans naturally make decisions. When we're trying to classify something, we don't use complex mathematical formulas - we ask simple questions one at a time. This makes classification trees incredibly interpretable and easy to understand, even for people without technical backgrounds.

## 12.1.1. The Three Key Components

### 1. Where to Split

This involves deciding on the variable (denoted as $`j`$) and the split value ($`s`$) that divides our data into two parts, based on whether $`X_j < s`$ or not.

**Intuition**: This is like choosing the best question to ask at each step. If you're trying to identify animals, you need to decide whether to ask about size, color, habitat, or some other feature, and what threshold to use. Should you ask "Is it bigger than a dog?" or "Is it bigger than a horse?" The choice of question and threshold determines how well you separate the different types of animals.

**Mathematical Formulation**: For a feature $`j`$ and split point $`s`$, we create two regions:
$$ R_1(j, s) = \{X | X_j \leq s\} \quad \text{and} \quad R_2(j, s) = \{X | X_j > s\} $$

**Key Considerations**:
- **Feature Selection**: Which variable provides the best split?
- **Split Point**: What threshold value maximizes separation?
- **Binary Splits**: Each split creates exactly two child nodes

**Intuition**: These considerations are like the strategic decisions in a game of "20 Questions":
- **Feature Selection**: Which question will give us the most information?
- **Split Point**: What's the best threshold for this question?
- **Binary Splits**: We can only ask yes/no questions, not multiple-choice

### 2. When to Stop

As previously discussed, the general strategy is to initially construct a large tree and then employ a pruning process based on a loss plus penalty criteria. This strategy helps prevent overfitting.

**Intuition**: This is like knowing when to stop asking questions. If you keep asking more and more specific questions, you might end up with rules that only work for the exact examples you've seen (overfitting). You need to know when to stop and make a decision based on what you've learned so far.

**Stopping Criteria**:
- **Minimum node size**: Stop when node contains fewer than $`n_{\min}`$ samples
- **Maximum depth**: Stop when tree reaches maximum depth $`d_{\max}`$
- **Pure nodes**: Stop when all samples in node belong to same class
- **Minimum improvement**: Stop when split improvement is below threshold

**Intuition**: These stopping criteria are like having rules for when to stop asking questions:
- **Minimum Node Size**: Don't ask more questions if you only have a few examples left
- **Maximum Depth**: Don't ask more than a certain number of questions
- **Pure Nodes**: Stop when you're confident about the classification
- **Minimum Improvement**: Stop when asking more questions doesn't help much

**Pruning Strategy**:
$$ \text{Cost}(T) = \text{Loss}(T) + \alpha \cdot \text{Complexity}(T) $$

where $`\alpha`$ is the regularization parameter controlling tree size.

**Intuition**: Pruning is like editing a decision tree to remove unnecessary questions. The cost function balances accuracy (loss) with simplicity (complexity). A high α means we prefer simpler trees, even if they're slightly less accurate.

### 3. How to Predict at Each Leaf Node

Depending on whether we are dealing with regression or classification, we adopt different approaches for making predictions at leaf nodes.

**Intuition**: This is like deciding what answer to give when you've finished asking questions. In regression, you might give an average value. In classification, you might give the most common class or a probability distribution over all classes.

#### Regression Trees
For regression, at each leaf node, we calculate the average Y value based on the training samples within that node:
$$ \hat{y}_{\text{leaf}} = \frac{1}{n_{\text{leaf}}} \sum_{i \in \text{leaf}} y_i $$

**Intuition**: This is like taking the average of all the examples that ended up at this leaf. If you asked "Is it bigger than a dog?" and "Does it have stripes?" and ended up with examples of tigers, lions, and leopards, you might predict the average size of these animals.

#### Classification Trees
For classification, we apply a similar concept. When a leaf node contains observations from $`K`$ classes, we can either:

**Majority Voting**:
$$ \hat{y}_{\text{leaf}} = \arg\max_{k} n_k $$

where $`n_k`$ is the number of samples of class $`k`$ in the leaf.

**Intuition**: This is like taking a vote among all the examples at this leaf. If you have 5 tigers, 2 lions, and 1 leopard, you predict "tiger" because it's the most common.

**Class Probabilities**:
$$ P(y = k | \text{leaf}) = \frac{n_k}{n_{\text{leaf}}} $$

where $`n_{\text{leaf}} = \sum_{k=1}^K n_k`$ is the total number of samples in the leaf.

**Intuition**: This is like giving probabilities instead of a single answer. In the same example, you might say "62.5% chance it's a tiger, 25% chance it's a lion, 12.5% chance it's a leopard."

## 12.1.2. Goodness-of-Split Criterion

### Regression vs Classification

In the context of regression, this often involves calculating the reduction in residual sum of squares. Specifically, we consider a node $`T`$:

**Regression Split Criterion**:
$$ \Delta \text{RSS} = \text{RSS}(T) - \left[\text{RSS}(T_L) + \text{RSS}(T_R)\right] $$

where:
- $`\text{RSS}(T) = \sum_{i \in T} (y_i - \bar{y}_T)^2`$
- $`\text{RSS}(T_L) = \sum_{i \in T_L} (y_i - \bar{y}_{T_L})^2`$
- $`\text{RSS}(T_R) = \sum_{i \in T_R} (y_i - \bar{y}_{T_R})^2`$

**Intuition**: In regression, we measure how much the split reduces the variance of the target variable. It's like asking "How much does this question help us predict the exact value?"

**Classification Split Criterion**:
For classification, we use impurity measures instead of RSS:

$$ \Delta I = I(T) - \left[\frac{n_L}{n_T} I(T_L) + \frac{n_R}{n_T} I(T_R)\right] $$

where $`I(T)`$ is the impurity measure for node $`T`$.

**Intuition**: In classification, we measure how much the split reduces the "confusion" or "mixedness" of the classes. It's like asking "How much does this question help us separate the different categories?"

### Common Impurity Measures

#### 1. Gini Impurity
$$ I_{\text{Gini}}(T) = 1 - \sum_{k=1}^K p_k^2 $$

where $`p_k = \frac{n_k}{n_T}`$ is the proportion of class $`k`$ in node $`T``.

**Intuition**: Gini impurity measures how "mixed" a node is. It's like measuring the probability that two randomly chosen examples from the node would be from different classes. A pure node (all same class) has Gini = 0, while a perfectly mixed node (equal proportions of all classes) has maximum Gini.

#### 2. Entropy
$$ I_{\text{Entropy}}(T) = -\sum_{k=1}^K p_k \log_2(p_k) $$

**Intuition**: Entropy measures the "information content" or "uncertainty" in a node. It's like asking "How much information do we need to specify which class a random example belongs to?" Pure nodes have zero entropy (no uncertainty), while mixed nodes have higher entropy (more uncertainty).

#### 3. Misclassification Error
$$ I_{\text{Error}}(T) = 1 - \max_k p_k $$

**Intuition**: Misclassification error is the simplest measure - it's just the probability of making a mistake if we predict the most common class. It's like asking "What fraction of examples would we get wrong if we always predict the majority class?"

### Properties of Impurity Measures

1. **Range**: All measures are in $`[0, 1]`$ for binary classification
2. **Minimum**: Achieved when node is pure (all samples same class)
3. **Maximum**: Achieved when classes are equally distributed
4. **Differentiability**: Gini and Entropy are differentiable, Error is not

**Intuition**: These properties ensure that our impurity measures behave sensibly:
- **Range**: All measures are on the same scale (0 to 1)
- **Minimum**: Pure nodes are the "best" (lowest impurity)
- **Maximum**: Perfectly mixed nodes are the "worst" (highest impurity)
- **Differentiability**: Smooth measures (Gini, Entropy) work better with optimization algorithms

## 12.1.3. The Greedy Algorithm

The process of searching for the best split follows a basic greedy algorithm:

**Intuition**: The greedy algorithm is like playing a game where you always make the best move you can see right now, without worrying too much about future consequences. It's like asking "What's the best question I can ask right now?" rather than "What's the best sequence of questions to ask?"

### Algorithm Steps

1. **Start at root node** with all training data
2. **For each feature** $`j = 1, 2, \ldots, p`$:
   - Sort unique values of feature $`j`$
   - **For each split point** $`s`$ (midpoint between consecutive values):
     - Split data: $`X_j \leq s`$ vs $`X_j > s`$
     - Calculate impurity reduction $`\Delta I`$
3. **Select best split**: Choose $`(j^*, s^*)`$ that maximizes $`\Delta I`$
4. **Create child nodes**: Split data according to best split
5. **Recurse**: Apply algorithm to each child node

**Intuition**: This algorithm is like systematically trying every possible question and choosing the best one:
- **Try Every Feature**: Consider asking about size, color, habitat, etc.
- **Try Every Threshold**: Consider "bigger than a mouse," "bigger than a cat," "bigger than a dog," etc.
- **Pick the Best**: Choose the question that gives the most information
- **Repeat**: Keep asking questions until you can't improve much more

### Computational Complexity

- **Time**: $`O(p \cdot n \log n)`$ per node (sorting dominates)
- **Space**: $`O(n)`$ for storing node data
- **Total**: $`O(p \cdot n \log n \cdot \text{number of nodes})`$

**Intuition**: The computational cost comes from:
- **Sorting**: We need to sort each feature to find good split points
- **Number of Features**: More features means more questions to try
- **Number of Nodes**: Bigger trees take longer to build

## 12.1.4. Implementation and Examples

The implementation of classification trees is provided in separate code files for both Python and R. These implementations demonstrate the core concepts of classification trees including impurity measures, tree building, and decision boundaries.

**Python Implementation**: The complete classification tree implementation is available in `code/introduction_implementation.py` and includes:
- **`ClassificationTree` class** with custom implementation of all impurity measures (Gini, Entropy, Misclassification Error) - like having a complete decision-making toolkit
- **Tree building algorithm** with recursive splitting and stopping criteria - like building a smart question-asking system
- **Impurity measure comparison** between Gini and Entropy criteria - like comparing different ways to measure confusion
- **Tree structure visualization** with different depths - like seeing how the decision tree grows
- **Stopping criteria demonstration** showing the effects of different parameters - like understanding when to stop asking questions
- **Greedy algorithm step-by-step demonstration** showing how splits are chosen - like watching the decision-making process
- **Advantages and limitations analysis** with different data patterns - like understanding when decision trees work well
- **Decision boundary visualization** and accuracy analysis - like seeing how the tree divides the data

**R Implementation**: The complete classification tree implementation is available in `code/r_introduction_implementation.R` and includes:
- **Basic tree demonstration** using rpart with tree visualization - like using professional decision-making tools
- **Impurity measures analysis** using Gini criterion (rpart default) - like understanding how confusion is measured
- **Tree structure analysis** with different depths and node counts - like analyzing decision tree complexity
- **Stopping criteria demonstration** with various parameter configurations - like testing different stopping rules
- **Greedy algorithm demonstration** showing split selection process - like watching the question-selection process
- **Advantages and limitations analysis** with different data patterns - like understanding practical constraints
- **Decision boundary visualization** using ggplot2 - like seeing how decisions are made

To run the classification tree demonstrations:

```python
# Python
from code.introduction_implementation import main
results = main()
```

```r
# R
source("code/r_introduction_implementation.R")
results <- main_r()
```

The implementations demonstrate how classification trees extend regression trees by using impurity measures instead of RSS, and how the greedy algorithm efficiently finds optimal splits to create interpretable decision boundaries.

## 12.1.5. Advantages and Limitations

### Advantages

1. **Interpretability**: Easy to understand and visualize
2. **No Assumptions**: No assumptions about data distribution
3. **Handles Mixed Data**: Can handle both numerical and categorical features
4. **Feature Importance**: Natural feature selection through splits
5. **Robust**: Insensitive to monotone transformations

**Intuition**: These advantages make classification trees like having a friendly, easy-to-understand decision-making system:
- **Interpretability**: You can literally follow the tree like a flowchart
- **No Assumptions**: Works with any type of data without needing to assume it follows a particular distribution
- **Mixed Data**: Can handle both numbers (like age) and categories (like color)
- **Feature Importance**: Naturally shows which features are most important for classification
- **Robust**: Works well even if you transform the data (like changing from pounds to kilograms)

### Limitations

1. **Instability**: Small changes in data can lead to very different trees
2. **Overfitting**: Tendency to overfit without proper regularization
3. **Axis-Aligned**: Can only create axis-aligned decision boundaries
4. **Greedy**: Local optimization may miss global optimum
5. **High Variance**: Individual trees have high variance

**Intuition**: These limitations are like the trade-offs of using a simple decision-making system:
- **Instability**: Like how a small change in the training data might completely change which questions you ask first
- **Overfitting**: Like memorizing the exact examples instead of learning general patterns
- **Axis-Aligned**: Can only ask questions like "Is X > 5?" not complex questions like "Is X + Y > 10?"
- **Greedy**: Like always choosing the best immediate move without considering long-term strategy
- **High Variance**: Like how different random samples of data might lead to very different decision trees

## 12.1.6. Summary

Classification trees extend regression trees to classification problems by:

1. **Impurity Measures**: Using Gini, entropy, or misclassification error instead of RSS
2. **Prediction Methods**: Majority voting or class probabilities at leaf nodes
3. **Split Criteria**: Maximizing impurity reduction
4. **Greedy Algorithm**: Same recursive splitting approach

**Intuition**: Classification trees are like upgrading from predicting numbers (regression) to predicting categories (classification). Instead of trying to predict the exact value, we're trying to predict which category something belongs to.

Key insights:
- **Impurity measures** control split quality - like choosing the best questions to ask
- **Stopping criteria** prevent overfitting - like knowing when to stop asking questions
- **Greedy approach** is computationally efficient - like making the best immediate decision
- **Tree structure** provides interpretability - like having a clear decision flowchart

This foundation sets the stage for more advanced tree-based methods like random forests and gradient boosting, which address many of the limitations of single classification trees.

**Intuition**: Classification trees are the building blocks for more sophisticated ensemble methods. While a single tree might be unstable or prone to overfitting, combining many trees (like in random forests) can create much more robust and accurate classifiers. It's like how a single expert might make mistakes, but a committee of experts is usually more reliable.

---

**Navigation:**
- **Next Topic:** [Impurity Measures](02_impurity_measures.md) - Mathematical foundations and properties of impurity measures for classification
- **Previous Topic:** [Classification Trees Overview](README.md) - Overview of classification trees and boosting algorithms
