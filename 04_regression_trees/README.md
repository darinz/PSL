# Regression Trees

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This module covers tree-based methods for regression, from single decision trees to ensemble methods including Random Forests and Gradient Boosting Machines (GBM). The content has been expanded and clarified for accessibility, with detailed mathematical derivations, code explanations, and improved formatting using inline ($`...`$) and display math (```math) LaTeX. Where possible, image-based equations and text have been converted to selectable, copyable LaTeX in the markdown files for clarity and accessibility.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand the fundamentals** of regression trees and their construction
- **Implement tree building algorithms** with proper splitting criteria and stopping rules
- **Apply pruning techniques** using cost complexity and cross-validation
- **Build ensemble methods** including Random Forests and Gradient Boosting
- **Interpret variable importance** measures from tree-based models
- **Handle categorical variables** and missing values in tree models
- **Implement these techniques** in both R and Python
- **Choose appropriate methods** for different regression scenarios

## Topics Covered

### 4.1 Regression Trees
- **Introduction to Regression Trees**: Understanding tree structure and terminology
- **Tree Building Process**: How to construct trees using recursive binary splitting
- **Splitting Criteria**: RSS-based splitting for regression problems
- **Handling Categorical Variables**: Efficient splitting strategies for categorical predictors
- **Missing Value Strategies**: Surrogate splits and other approaches
- **Pruning Techniques**: Cost complexity pruning and the Weakest Link Algorithm
- **Cross-Validation**: Selecting optimal complexity parameters
- **Expanded mathematical derivations and LaTeX formatting**

**Key Concepts:**
- Recursive binary partitioning
- RSS-based splitting criteria
- Cost complexity trade-off
- Tree visualization and interpretation
- Stopping criteria and overfitting prevention

### 4.2 Random Forest
- **Introduction to Random Forest**: Ensemble methods and bagging principles
- **Bootstrap Sampling**: Understanding bootstrap samples and out-of-bag (OOB) samples
- **Bagging Algorithm**: Bootstrap aggregation for reducing variance
- **Random Forest Constraints**: Feature subsetting to decorrelate trees
- **Variable Importance**: RSS gain and permutation-based importance measures
- **Performance Evaluation**: OOB error estimation and model assessment
- **Expanded code and math explanations**

**Key Concepts:**
- Bootstrap sampling with replacement
- Out-of-bag (OOB) samples for validation
- Feature subsetting (m = √p for classification, m = p/3 for regression)
- Ensemble diversity and decorrelation
- Variable importance interpretation

### 4.3 Gradient Boosting Machines (GBM)
- **Introduction to Boosting**: Forward stagewise additive modeling
- **Boosting Algorithm**: Iterative fitting of weak learners to residuals
- **Tuning Parameters**: Learning rate, number of trees, and tree complexity
- **Regularization**: Subsampling and shrinkage to prevent overfitting
- **Performance Monitoring**: Cross-validation and early stopping
- **Comparison with Random Forest**: When to use each method
- **Expanded code and LaTeX math explanations**

**Key Concepts:**
- Forward stagewise optimization
- Residual fitting and additive models
- Learning rate and shrinkage
- Subsampling for regularization
- Overfitting prevention strategies

## Code Examples

### Python Implementation

#### Regression Trees (`Python_W4_RegressionTree.py`)
```python
# Fit regression tree with cost complexity pruning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Fit initial tree
tr1 = DecisionTreeRegressor(max_leaf_nodes=10)
tr1.fit(X, y)

# Cost complexity pruning path
path = tr1.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Cross-validation for optimal alpha
cv = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(clf, X, y, cv=cv)
```

#### Random Forest (`Python_W4_Regression_RandomForest.py`)
```python
# Fit Random Forest with OOB scoring
from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(
    n_estimators=400, 
    oob_score=True, 
    max_features=1.0/3
)
rfModel.fit(X_train, y_train)

# Variable importance
importances = rfModel.feature_importances_
```

#### Gradient Boosting (`Python_W4_Regression_GBM.py`)
```python
# Fit Gradient Boosting with different parameters
from sklearn.ensemble import GradientBoostingRegressor

# High learning rate, few trees
model1 = GradientBoostingRegressor(
    learning_rate=1.0, 
    n_estimators=100
)

# Low learning rate, many trees, subsampling
model2 = GradientBoostingRegressor(
    learning_rate=0.02, 
    n_estimators=1000, 
    subsample=0.5
)
```

### R Implementation

#### Regression Trees (`Rcode_W4_RegressionTree.R`)
- Tree construction using `rpart` package
- Cost complexity pruning with `prune.rpart`
- Cross-validation for optimal complexity parameter
- Tree visualization and interpretation

#### Random Forest (`Rcode_W4_Regression_RandomForest.R`)
- Random Forest implementation with `randomForest`
- OOB error estimation and performance monitoring
- Variable importance analysis
- Model comparison and evaluation

#### Gradient Boosting (`Rcode_W4_Regression_GBM.R`)
- GBM implementation with `gbm` package
- Parameter tuning and cross-validation
- Performance monitoring and early stopping
- Feature importance analysis

## Key Mathematical Concepts

### Tree Building
- **Splitting Criterion**: $`\text{RSS} = \sum (y_i - \bar{y}_L)^2 + \sum (y_i - \bar{y}_R)^2`$
- **Prediction at Leaf**: $`\bar{y}_{\text{leaf}} =`$ average of responses in leaf node
- **Cost Complexity**: $`R_\alpha(T) = \text{RSS}(T) + \alpha|T|`$

### Random Forest
- **Bootstrap Sample**: Sample n observations with replacement
- **OOB Sample**: Observations not in bootstrap sample
- **Feature Subsetting**: Consider $`m = p/3`$ features at each split
- **Final Prediction**: Average predictions from all trees

### Gradient Boosting
- **Additive Model**: $`F(x) = f_1(x) + f_2(x) + ... + f_T(x)`$
- **Residual Fitting**: Fit tree to residuals $`r_i = y_i - F_{t-1}(x_i)`$
- **Shrinkage**: $`F_t(x) = F_{t-1}(x) + \eta \cdot f_t(x)`$
- **Subsampling**: Use fraction of training data for each tree

## Practical Applications

### When to Use Each Method

1. **Single Regression Trees**:
   - Need for interpretable models
   - Small to moderate datasets
   - Understanding variable interactions
   - Initial exploration of data structure

2. **Random Forest**:
   - High-dimensional data
   - Need for robust predictions
   - Variable importance assessment
   - Handling missing values automatically
   - Less tuning required

3. **Gradient Boosting**:
   - Maximum prediction accuracy
   - Computational resources available
   - Willing to tune multiple parameters
   - Sequential learning preferred

### Data Characteristics

- **Categorical Variables**: Tree methods handle these naturally
- **Missing Values**: Multiple strategies available (surrogate splits, OOB)
- **Non-linear Relationships**: Trees capture these automatically
- **Variable Interactions**: Trees model interactions implicitly
- **Scale Invariance**: No need for feature scaling

## Model Comparison

### Advantages of Tree-Based Methods
- **Interpretability**: Easy to understand and explain
- **Automatic Feature Selection**: Built-in variable importance
- **Handles Mixed Data Types**: Categorical and numerical variables
- **Robust to Outliers**: Less sensitive than linear methods
- **Captures Non-linearities**: No need for feature engineering

### Limitations
- **High Variance**: Individual trees can be unstable
- **Black Box**: Ensemble methods less interpretable
- **Computational Cost**: Training can be expensive
- **Overfitting Risk**: Especially with deep trees

## Visualization and Analysis

The code examples include comprehensive visualizations:

- **Tree Structure**: Visual representation of decision trees
- **Pruning Paths**: Cost complexity vs tree size
- **Cross-Validation Plots**: Model selection using CV
- **Performance Monitoring**: Training/test error over iterations
- **Variable Importance**: Feature ranking and importance scores
- **OOB Error Plots**: Random Forest performance vs number of trees

## Getting Started

1. **Review the theoretical foundations** in the markdown files
2. **Start with single regression trees** to understand the basics
3. **Experiment with Random Forest** for robust predictions
4. **Try Gradient Boosting** for maximum performance
5. **Compare methods** on your own datasets
6. **Practice parameter tuning** and model selection
7. **Reference expanded math/code explanations and LaTeX formatting throughout**

## Additional Resources

- **Textbooks**: Elements of Statistical Learning (ESL), Introduction to Statistical Learning (ISL)
- **Papers**: Original Random Forest and GBM papers
- **Software**: scikit-learn, randomForest, gbm, xgboost packages
- **Online Resources**: Statistical learning course materials and tutorials

## Contributing

Feel free to contribute improvements to the code examples or documentation. This module is designed to be a comprehensive resource for learning tree-based regression methods in statistical learning. 