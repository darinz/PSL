# Boosting

## Simple (weak) classifiers are good!

Weak classifiers, despite their simplicity, offer several advantages that make them valuable building blocks for more sophisticated learning algorithms.

### Examples of Weak Classifiers

<img src="./img/04_simple_classifers.png" width="500px">

**Logistic Regression with Simple Features:**
A linear classifier that creates a diagonal decision boundary to separate classes. In a 2D scatter plot with purple and black data points, logistic regression with simple features creates a diagonal green line that provides a basic but effective separation.\
**Shallow Decision Trees:**
Small tree structures with limited depth, typically containing:
- A root node
- Two intermediate nodes
- Four leaf nodes (colored purple, blue, gray, and green/orange)

These shallow trees make simple, interpretable decisions without the complexity of deep trees.

**Decision Stumps:**
The simplest form of decision trees - single-level trees with one split. For example:
- **Split Question:** "Income > $100K?"
- **Outcomes:** 
  - **Yes:** Predict Safe (green oval)
  - **No:** Predict Risky (orange oval)

Decision stumps represent the most basic form of decision-making in tree-based models.

### Characteristics of Weak Classifiers

**Advantages:**
- **Low variance:** Predictions are stable and consistent
- **Learning is fast:** Training time is minimal due to simplicity
- **Interpretable:** Easy to understand and explain
- **Robust:** Less prone to overfitting on small datasets

**Disadvantages:**
- **High bias:** May underfit complex patterns in the data
- **Limited expressiveness:** Cannot capture complex non-linear relationships
- **Poor performance on complex datasets:** May achieve only slightly better than random performance

## Finding a classifier that's just right

The challenge in machine learning is finding the optimal balance between model complexity and performance, navigating the bias-variance trade-off.

### The Bias-Variance Trade-off

**Model Complexity vs. Classification Error:**

<img src="./img/04_classifer_just_right.png" width="400px">

As model complexity increases, we observe different behaviors in training and true error:

**Training Error (Purple Curve):**
- Starts high for simple models
- Rapidly decreases as complexity increases
- Eventually flattens out at a low error value
- Continues to decrease with additional complexity

**True Error (Green Curve):**
- Starts high for simple models (high bias)
- Decreases initially as complexity increases
- Reaches a minimum point (marked by orange star)
- Then increases again due to overfitting (high variance)
- Forms a U-shaped curve

**Optimal Complexity:**
The sweet spot lies at the minimum of the true error curve, where we achieve the best generalization performance.

### Options for Improvement

When faced with the challenge of improving classifier performance, we have two main approaches:

**Option 1: Add more features or depth**
- Increase model complexity by adding more features
- Use deeper decision trees
- Employ more sophisticated algorithms
- **Risk:** May lead to overfitting and increased variance

**Option 2: ?????**
- What are other options and alternative approach available?
- There might be a different strategy beyond simply increasing complexity

## Boosting question

The fundamental question that led to the development of boosting algorithms was whether multiple weak learners could be combined to create a stronger, more effective classifier.

### The Research Question

**Can a set of weak learners be combined to create a stronger learner?**

This question was first formally posed by Kearns and Valiant in 1988, setting the foundation for theoretical work in ensemble learning.

### The Answer

**Yes!** Schapire (1990) provided the theoretical foundation and practical algorithm that demonstrated how weak learners could indeed be combined to create a stronger learner.

### The Concept: Boosting

**Boosting** is an ensemble learning technique that combines multiple weak classifiers to create a strong classifier. The key insight is that by carefully weighting and combining the predictions of multiple simple models, we can achieve better performance than any individual weak learner.

### Amazing Impact

Boosting has had a transformative impact on machine learning and data science:

**Simple Approach:**
- Conceptually straightforward to understand
- Easy to implement and apply
- Based on intuitive principles of learning from mistakes

**Widely Used in Industry:**
- Applied across numerous domains and applications
- Standard tool in many machine learning pipelines
- Proven track record in production systems

**Wins Most Kaggle Competitions:**
- Dominates competitive machine learning
- Consistently achieves top performance
- Preferred choice for structured data problems

**Great Systems:**
- **XGBoost:** Extreme Gradient Boosting, one of the most popular implementations
- **LightGBM:** Microsoft's gradient boosting framework
- **CatBoost:** Yandex's gradient boosting library
- **AdaBoost:** The original boosting algorithm

The success of boosting demonstrates that sometimes the best approach isn't to make individual models more complex, but rather to intelligently combine multiple simple models. This insight has fundamentally changed how we think about machine learning and has led to some of the most powerful and widely-used algorithms in the field.

## Ensemble Classifier

### Single Classifier: The Building Block

A single classifier, often referred to as a "weak learner" in the context of ensemble methods, takes an input and produces a prediction. It serves as the fundamental unit that ensemble methods combine to form a more robust model.

**Input and Output Flow:**
The process of a single classifier can be visualized as a simple decision flow:

1. **Input:** An input vector $x$ (e.g., loan application data)
2. **Decision Node:** A single decision rule is applied (e.g., `Income > $100K?`)
3. **Output:** Based on the decision, a classification $\hat{y} = f(x)$ is produced. This output is typically binary, such as:
   - `+1` (e.g., "Safe" loan)
   - `-1` (e.g., "Risky" loan)

**Example: Loan Application Classifier**
Consider a simple classifier for loan applications:

<img src="./img/04_single_classifier.png" width="400px">

- **Input:** A loan application $x$
- **Decision:** Is the applicant's `Income > $100K`?
  - If `Yes`, the loan is classified as **Safe**
  - If `No`, the loan is classified as **Risky**

This simple classifier, represented by a single decision node, provides a basic prediction for the input $x$.

### Ensemble Methods: Combining Weak Classifiers

Ensemble methods combine the predictions of multiple individual classifiers (often weak learners) to produce a more accurate and robust final prediction. Each individual classifier "votes" on the prediction, and these votes are aggregated.

**Example: Multiple Classifiers Voting on a Loan Application**
Let's consider a specific loan application $x = (\text{Income}=\$120K, \text{Credit}=\text{Bad}, \text{Savings}=\$50K, \text{Market}=\text{Good})$. We use four different weak classifiers, each focusing on a different feature:

<img src="./img/04_ensemble_methods.png" width="600px">

1. **Classifier 1 ($f_1(x)$): Income > $100K?**
   - Input: $x$ (Income=$120K$)
   - Decision: Yes
   - Output: Safe ($f_1(x) = +1$)

2. **Classifier 2 ($f_2(x)$): Credit history?**
   - Input: $x$ (Credit=Bad)
   - Decision: Bad
   - Output: Risky ($f_2(x) = -1$)

3. **Classifier 3 ($f_3(x)$): Savings > $100K?**
   - Input: $x$ (Savings=$50K$)
   - Decision: No
   - Output: Risky ($f_3(x) = -1$)

4. **Classifier 4 ($f_4(x)$): Market conditions?**
   - Input: $x$ (Market=Good)
   - Decision: Good
   - Output: Safe ($f_4(x) = +1$)

**Combining Predictions: The Ensemble Model**
To combine these individual predictions, an ensemble model learns coefficients (weights) for each classifier. The final prediction is a weighted sum of the individual classifier outputs, passed through a sign function for binary classification:

$$F(x_i) = \text{sign}(w_1 f_1(x_i) + w_2 f_2(x_i) + w_3 f_3(x_i) + w_4 f_4(x_i))$$

Here, $w_j$ represents the learned coefficient (weight) for classifier $f_j(x_i)$. The `sign` function converts the weighted sum into a binary output (+1 or -1).

### Ensemble Classifier in General

An ensemble classifier aims to leverage the collective intelligence of multiple individual classifiers to achieve superior performance compared to any single classifier.

**Goal:**
- **Predict output $y$:** The target variable, typically binary (`+1` or `-1`)
- **From input $x$:** The feature vector representing the data point

**Learn Ensemble Model:**
The learning process for an ensemble model involves two key components:

1. **Classifiers:** A set of $T$ individual classifiers, denoted as $f_1(x), f_2(x), \dots, f_T(x)$. These are often weak learners.
2. **Coefficients:** A set of learned weights (or coefficients) for each classifier, denoted as $\hat{w}_1, \hat{w}_2, \dots, \hat{w}_T$. These coefficients determine the influence of each classifier on the final prediction.

**Prediction:**
The final prediction $\hat{y}$ from an ensemble classifier is given by the weighted sum of the individual classifier predictions, passed through a sign function:

$$\hat{y} = \text{sign} \left( \sum_{t=1}^{T} \hat{w}_t f_t(x) \right)$$

This formula represents the core mechanism of many boosting algorithms, where weak classifiers are iteratively trained and combined with learned weights to form a strong ensemble model.

## Boosting
