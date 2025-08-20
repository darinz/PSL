# Decision Trees - Predicting potential loan defaults

What makes a loan risky?

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

