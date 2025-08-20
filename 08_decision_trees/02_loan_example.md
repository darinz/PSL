# Decision Trees - Predicting potential loan defaults

## What Makes a Loan Risky?

When applying for a loan to buy a new house, several key factors determine the risk assessment. The loan application process evaluates multiple aspects of the borrower's financial situation and personal circumstances.

**Credit History** is one of the most critical factors. Lenders examine whether the borrower has paid previous loans on time, looking for patterns of responsible financial behavior. Credit history is typically categorized as excellent, good, or fair based on past payment records and credit utilization.

**Income** represents the borrower's earning capacity and ability to make regular loan payments. Lenders assess the stability and amount of income, such as an annual salary of $80,000, to determine if the borrower can comfortably afford the loan payments.

**Loan Terms** specify the repayment timeline and conditions. This includes how soon the loan needs to be paid back, with common terms being 3 years, 5 years, or longer periods depending on the loan type and amount.

**Personal Information** encompasses various demographic and situational factors including age, the reason for the loan, marital status, and other relevant details. For example, a home loan application for a married couple would be evaluated differently than a loan for a single individual.

## Classifier Review

<img src="./img/02_classifier.png" width="500px">

In the context of loan applications, a classifier model takes loan application data as input and produces a predicted class as output. The process can be summarized as:

- **Input:** $x_i$ (Loan Application data)
- **Model:** Classifier MODEL
- **Output:** $\hat{y}_i$ (Predicted class)

The classifier produces two possible outcomes:
- **Safe** ($\hat{y}_i = +1$): Loan is approved
- **Risky** ($\hat{y}_i = -1$): Loan is denied

## Decision Tree for Loan Applications

A decision tree provides an intuitive way to classify loan applications based on multiple criteria. Here's an example decision tree structure:

<img src="./img/02_loan_decision_tree.png" width="500px">

### Decision Tree Structure

The decision tree evaluates loan applications through a series of questions:

1. **Credit?** (First decision point)
   - **excellent** → **Safe** (immediate approval)
   - **fair** → Continue to Term evaluation
   - **poor** → Continue to Income evaluation

2. **Term?** (for fair credit)
   - **3 years** → **Risky**
   - **5 years** → **Safe**

3. **Income?** (for poor credit)
   - **high** → Continue to Term evaluation
   - **low** → **Risky**

4. **Term?** (for poor credit, high income)
   - **3 years** → **Risky**
   - **5 years** → **Safe**

### Example: Scoring a Loan Application

<img src="./img/02_scoring.png" width="600px">

Consider a loan application with the following characteristics:
- **Credit:** poor
- **Income:** high  
- **Term:** 5 years

**Decision Path:**
1. Start → Credit = poor → Income evaluation
2. Income = high → Term evaluation
3. Term = 5 years → **Safe**

**Result:** $\hat{y}_i = \text{Safe}$ (Loan approved)

This decision tree demonstrates how multiple factors are considered in sequence to arrive at a final classification decision, with the path through the tree determined by the specific values of the input features.

## Decision Tree Learning Task

### The Learning Problem

The decision tree learning problem involves constructing an optimal decision tree from training data. Given a dataset of $N$ observations $(x_i, y_i)$, where each $x_i$ represents the features of a loan application and $y_i$ is the corresponding classification (safe or risky), the goal is to learn a decision tree function $T(X)$ that accurately predicts the loan risk.

<img src="./img/02_training_data.png" width="500px">

**Training Data Structure:**
The training dataset contains examples with the following features:
- **Credit:** excellent, fair, poor
- **Term:** 3 yrs, 5 yrs  
- **Income:** high, low
- **Target (y):** safe, risky

The learning process optimizes a quality metric on the training data to find the best decision tree structure.

### Quality Metric: Classification Error

To evaluate the performance of a decision tree, we use classification error as the primary quality metric.

**Definition:** Error measures the fraction of mistakes made by the classifier.

**Formula:**

$$\text{Error} = \frac{\text{num incorrect predictions}}{\text{num examples}}$$

**Value Range:**
- **Best possible value:** 0.0 (perfect classification)
- **Worst possible value:** 1.0 (all predictions incorrect)

The classification error provides a straightforward measure of how well the decision tree performs on the training data, with lower values indicating better performance.

### The Challenge of Finding the Best Tree

Decision tree learning presents a significant computational challenge due to the exponentially large number of possible tree configurations.

**Complexity Problem:**
The space of possible decision trees grows exponentially with the number of features and their possible values. For any given dataset, there are numerous valid tree structures that could be constructed, each with different branching patterns and decision rules.

**NP-Hard Problem:**
Learning the smallest (most parsimonious) decision tree that achieves optimal performance is an NP-hard problem, as proven by Hyafil & Rivest in 1976. This means that finding the globally optimal decision tree is computationally intractable for realistic problem sizes.

**Multiple Valid Solutions:**

<img src="./img/02_trees.png" width="500px">

Given the same training data, multiple different tree structures (denoted as $T_1(X)$ through $T_6(X)$ in the illustration) can achieve similar or identical performance on the training set. Each tree may have different:

- Root node choices
- Branching patterns
- Decision thresholds
- Overall complexity

This inherent complexity necessitates the use of heuristic algorithms and greedy approaches to construct decision trees efficiently, trading optimality for computational feasibility.

## Greedy Decision Tree Learning - Training Data and Initial Representation

### Our Training Data Table

To illustrate the decision tree learning process, we use a training dataset with $N = 40$ examples and 3 features. The dataset contains loan application information with the following structure:

**Dataset Specifications:**
- **Size:** $N = 40$ observations
- **Features:** 3 (Credit, Term, Income)
- **Target:** Binary classification (safe/risky)

**Example Training Data:**
| Credit    | Term   | Income | y      |
|-----------|--------|--------|--------|
| excellent | 3 yrs  | high   | safe   |
| fair      | 5 yrs  | low    | risky  |
| fair      | 3 yrs  | high   | safe   |
| poor      | 5 yrs  | high   | risky  |
| excellent | 3 yrs  | low    | risky  |
| fair      | 5 yrs  | low    | safe   |
| poor      | 3 yrs  | high   | risky  |
| poor      | 5 yrs  | low    | safe   |
| fair      | 3 yrs  | high   | safe   |
| ...       | ...    | ...    | ...    |

The dataset contains a mix of loan applications with varying credit histories, loan terms, and income levels, each labeled as either safe or risky based on historical outcomes.

### Start with All the Data

When beginning the decision tree construction process, we start with the complete dataset and examine the overall distribution of loan outcomes.

<img src="./img/02_all_data.png" width="400px">

**Initial Data Distribution:**
- **Safe loans:** 22 examples
- **Risky loans:** 18 examples
- **Total examples:** $N = 40$

This initial state represents the root node of our decision tree, where all training examples are grouped together without any feature-based splitting.

### Compact Visual Notation: Root Node

<img src="./img/02_root_node.png" width="400px">

The root node can be represented using a compact visual notation that summarizes the class distribution:

**Root Node Representation:**
```
[22, 18]
```

Where:
- **22** (green) represents the number of safe loans
- **18** (red) represents the number of risky loans

This notation provides a concise way to represent the current state of the data at any node in the decision tree, making it easy to track how the data distribution changes as we apply different splitting criteria.

The root node serves as the starting point for the greedy decision tree learning algorithm, from which we will iteratively apply feature-based splits to create a hierarchical classification structure.

### Decision Stump: Single Level Tree

<img src="./img/02_stump.png" width="400px">

A decision stump represents the simplest form of a decision tree - a single-level tree with one split. Starting from the root node containing all data `[22, 18]`, we apply a single feature-based split to create a decision stump.

**Decision Stump Structure:**
The decision stump begins with the root node and applies a single split based on the Credit feature:

- **Root Node:** `[22, 18]` (all data)
- **Split Feature:** Credit
- **Split Outcomes:**
  - **excellent:** `[9, 0]` - Subset of data with Credit = excellent
  - **fair:** `[9, 4]` - Subset of data with Credit = fair  
  - **poor:** `[4, 14]` - Subset of data with Credit = poor

This single split partitions the original dataset into three distinct subsets based on credit history, with each subset containing a different distribution of safe and risky loans.

### Visual Notation: Intermediate Nodes

<img src="./img/02_intermediate.png" width="400px">

The nodes created by the split are called intermediate nodes, as they represent subsets of data that could potentially be split further in a more complex tree structure.

**Intermediate Node Representation:**

Each intermediate node shows the class distribution for its corresponding data subset:

- **excellent credit node:** `[9, 0]` - All excellent credit applications are safe
- **fair credit node:** `[9, 4]` - Fair credit applications are mostly safe
- **poor credit node:** `[4, 14]` - Poor credit applications are mostly risky

These intermediate nodes serve as the foundation for building more complex decision trees, where each node could potentially be split further based on additional features.

### Making Predictions with a Decision Stump

<img src="./img/02_predictions.png" width="550px">

To make predictions using a decision stump, we apply a simple majority rule at each intermediate node.

**Prediction Rule:**
For each intermediate node, set $\hat{y}$ = majority value

**Predictions by Credit Category:**
- **excellent credit:** `[9, 0]` → **Safe** (majority: 9 safe vs 0 risky)
- **fair credit:** `[9, 4]` → **Safe** (majority: 9 safe vs 4 risky)
- **poor credit:** `[4, 14]` → **Risky** (majority: 4 safe vs 14 risky)

**Decision Stump Classification:**
The decision stump creates a simple classification rule:
- If Credit = excellent → Predict Safe
- If Credit = fair → Predict Safe  
- If Credit = poor → Predict Risky

This demonstrates how even a simple single-level decision tree can capture meaningful patterns in the data, with credit history serving as a strong predictor of loan risk. The decision stump provides a baseline model that can be extended to more complex tree structures by adding additional splits at the intermediate nodes.

## Selecting Best Feature to Split on

