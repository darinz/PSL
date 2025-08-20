# Decision Trees - Predicting potential loan defaults

What makes a loan risky?

I want to buy a new house! Loan Application
- Credit History
- Income
- Term
- Personal Info

Credit History Explained
- Did I pay previous loans on time?
- Example: excellent, good, or fair

Income
- What's my income?
- Example: $80K per year

Loan terms
- How soon do I need to pay the loan?
- Example: 3 years, 5 years, ...

Personal information
- Age, reason for the loan, marital status, ...
- Example: Home loan for a married couple

## Classifier Review

<img src="./img/02_classifier.png" width="500px">

In the context of loan applications, a classifier model takes loan application data as input and produces a predicted class as output. The process can be summarized as:

**Input:** $x_i$ (Loan Application data)
**Model:** Classifier MODEL
**Output:** $\hat{y}_i$ (Predicted class)

The classifier produces two possible outcomes:
- **Safe** ($\hat{y}_i = +1$): Loan is approved
- **Risky** ($\hat{y}_i = -1$): Loan is denied

## Decision Tree for Loan Applications

A decision tree provides an intuitive way to classify loan applications based on multiple criteria. Here's an example decision tree structure:

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

