# Variable Selection and Regularization

This module covers essential techniques for variable selection and regularization in statistical learning, focusing on methods to handle high-dimensional data and improve model performance through feature selection and coefficient shrinkage.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand the motivation** behind variable selection and regularization
- **Compare different model selection criteria** (AIC, BIC, Mallow's Cp)
- **Implement subset selection methods** (best subset, forward, backward, stepwise)
- **Apply regularization techniques** (Ridge and Lasso regression)
- **Interpret the geometric and optimization perspectives** of regularization
- **Choose appropriate methods** for different data scenarios
- **Implement these techniques** in both R and Python

## Topics Covered

### 3.1 Subset Selection
- **Why Subset Selection**: Understanding the bias-variance trade-off and the need for variable selection
- **Selection Criteria**: AIC, BIC, and Mallow's Cp for model comparison
- **AIC vs BIC**: Philosophical differences and practical implications
- **Search Algorithms**: Level-wise search, greedy algorithms (forward, backward, stepwise)
- **Variable Screening**: Handling cases where p > n

**Key Concepts:**
- Training vs test error decomposition
- Model complexity penalties
- Computational efficiency considerations
- Variable screening for high-dimensional data

### 3.2 Regularization Framework
- **Unified Objective Function**: Framing variable selection as optimization problems
- **L0, L1, and L2 Penalties**: Understanding different regularization approaches
- **Data Preprocessing**: Centering and scaling for consistent results
- **Scale Invariance**: Ensuring methods work regardless of variable scaling

**Key Concepts:**
- Regularization as constrained optimization
- Importance of data standardization
- Geometric interpretation of penalties

### 3.3 Ridge Regression
- **Introduction**: L2 penalty and quadratic optimization
- **Shrinkage Effect**: Understanding coefficient shrinkage in orthogonal and non-orthogonal cases
- **SVD Perspective**: Using singular value decomposition to understand ridge behavior
- **Degree of Freedom**: Effective degrees of freedom and model complexity

**Key Concepts:**
- Ridge regression as shrinkage method
- Relationship between λ and effective degrees of freedom
- Bias-variance trade-off in ridge regression

### 3.4 Lasso Regression
- **Introduction**: L1 penalty and sparse solutions
- **Soft Thresholding**: Understanding the one-dimensional lasso solution
- **Lasso vs Ridge**: Geometric and optimization perspectives
- **Coordinate Descent**: Algorithm for solving lasso problems
- **Uniqueness**: Conditions for unique lasso solutions

**Key Concepts:**
- Variable selection through coefficient sparsity
- Soft thresholding operator
- Geometric interpretation with L1 ball constraints
- Computational algorithms for lasso

### 3.5 Discussion and Comparison
- **Method Selection**: When to use each approach
- **Simulation Studies**: Comparing methods on different data scenarios
- **Practical Guidelines**: Choosing methods based on data characteristics

## Code Examples

### Python Implementation

#### Subset Selection (`Python_W3_VarSel_SubsetSelection.py`)
```python
# Best subset selection with AIC/BIC
def bestsubset(X, Y):
    """Find best subset for each model size"""
    # Implementation details...

# Stepwise selection
def stepAIC(X, Y, features=None, AIC=True):
    """Stepwise selection based on AIC or BIC"""
    # Implementation details...
```

#### Ridge and Lasso (`Python_W3_VarSel_RidgeLasso.py`)
```python
# Ridge regression with cross-validation
ridgecv = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error')
ridgecv.fit(X_train, Y_train)

# Lasso regression with cross-validation
lassocv = LassoCV(alphas=alphas, cv=10)
lassocv.fit(X_train, Y_train)
```

### R Implementation

#### Subset Selection (`R_W3_VarSel_SubsetSelection.R`)
- Best subset selection using `leaps` package
- AIC/BIC model selection
- Stepwise selection procedures

#### Ridge and Lasso (`Rcode_W3_VarSel_RidgeLasso.R`)
- Ridge regression with `glmnet`
- Lasso regression with cross-validation
- Coefficient path analysis
- Model comparison and evaluation

## Key Mathematical Concepts

### Model Selection Criteria
- **AIC**: `-2log(likelihood) + 2p`
- **BIC**: `-2log(likelihood) + log(n)p`
- **Mallow's Cp**: `RSS + 2σ²p`

### Regularization Objectives
- **Ridge**: `min ||y - Xβ||² + λ||β||²`
- **Lasso**: `min ||y - Xβ||² + λ||β||₁`

### Soft Thresholding (Lasso)
```
β̂_j^lasso = sign(β̂_j^LS)(|β̂_j^LS| - λ/2)_+
```

## Practical Applications

### When to Use Each Method

1. **Subset Selection**:
   - Small to moderate number of predictors (p < 40)
   - Need for interpretable models
   - Computational resources available

2. **Ridge Regression**:
   - Multicollinearity present
   - All variables potentially relevant
   - Need for stable coefficient estimates

3. **Lasso Regression**:
   - High-dimensional data (p > n)
   - Sparse solutions desired
   - Variable selection needed

### Data Scenarios

- **X1**: Small, curated feature set → Full model often sufficient
- **X2**: Correlated but relevant features → Ridge/PCR effective
- **X3**: Many noise features → Lasso preferred for variable selection

## Visualization and Analysis

The code examples include comprehensive visualizations:

- **Coefficient Paths**: How coefficients change with regularization strength
- **Cross-Validation Paths**: Model selection using CV
- **Model Comparison Plots**: AIC/BIC vs model size
- **Performance Evaluation**: Test MSE comparisons

## Getting Started

1. **Review the theoretical foundations** in the markdown files
2. **Run the Python examples** to understand implementation
3. **Experiment with the R code** for additional insights
4. **Try the simulation study** to compare methods
5. **Apply to your own datasets** to gain practical experience

## Additional Resources

- **Textbooks**: Elements of Statistical Learning (ESL)
- **Papers**: Original lasso and ridge regression papers
- **Software**: scikit-learn, glmnet, leaps packages
- **Online Resources**: Statistical learning course materials

## Contributing

Feel free to contribute improvements to the code examples or documentation. This module is designed to be a comprehensive resource for learning variable selection and regularization techniques in statistical learning. 