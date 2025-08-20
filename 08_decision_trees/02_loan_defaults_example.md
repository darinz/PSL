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

$$\text{Error} = \frac{\text{\# incorrect predictions}}{\text{\# examples}}$$

**Value Range:**
- **Best possible value:** 0.0 (perfect classification)
- **Worst possible value:** 1.0 (all predictions incorrect)

The classification error provides a straightforward measure of how well the decision tree performs on the training data, with lower values indicating better performance.

### The Challenge of Finding the Best Tree

Decision tree learning presents a significant computational challenge due to the exponentially large number of possible tree configurations.

<img src="./img/02_trees.png" width="500px">

**Complexity Problem:**
The space of possible decision trees grows exponentially with the number of features and their possible values. For any given dataset, there are numerous valid tree structures that could be constructed, each with different branching patterns and decision rules.

**NP-Hard Problem:**
Learning the smallest (most parsimonious) decision tree that achieves optimal performance is an NP-hard problem, as proven by Hyafil & Rivest in 1976. This means that finding the globally optimal decision tree is computationally intractable for realistic problem sizes.

**Multiple Valid Solutions:**
Given the same training data, multiple different tree structures (denoted as $T_1(X)$ through $T_6(X)$ in the illustration) can achieve similar or identical performance on the training set. Each tree may have different:
- Root node choices
- Branching patterns
- Decision thresholds
- Overall complexity

This inherent complexity necessitates the use of heuristic algorithms and greedy approaches to construct decision trees efficiently, trading optimality for computational feasibility.

