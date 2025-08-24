# Decision Trees Code Examples

This folder contains comprehensive Python and R code examples for the Decision Trees chapter, covering all major concepts and techniques.

## File Structure

### Python Examples

- **`decision_trees_introduction.py`** - Fundamental concepts of decision trees
- **`loan_example.py`** - Loan application classification using decision trees
- **`overfitting.py`** - Overfitting demonstration and regularization techniques
- **`boosting.py`** - AdaBoost and ensemble boosting methods
- **`ensemble_methods.py`** - Bagging, Random Forests, and ensemble techniques

### R Examples

- **`decision_trees_introduction.R`** - Fundamental concepts of decision trees
- **`loan_example.R`** - Loan application classification using decision trees
- **`overfitting.R`** - Overfitting demonstration and regularization techniques
- **`boosting.R`** - AdaBoost and ensemble boosting methods
- **`ensemble_methods.R`** - Bagging, Random Forests, and ensemble techniques

### Original Files

- **`boosting_example.m`** - Original MATLAB example for boosting demonstration

## Installation Requirements

### Python Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### R Dependencies

```r
install.packages(c("rpart", "rpart.plot", "ggplot2", "dplyr", "caret", 
                   "randomForest", "adabag", "gridExtra", "pROC"))
```

## Usage

### Running Python Examples

Each Python file can be run independently:

```bash
python decision_trees_introduction.py
python loan_example.py
python overfitting.py
python boosting.py
python ensemble_methods.py
```

### Running R Examples

Each R file can be run independently:

```bash
Rscript decision_trees_introduction.R
Rscript loan_example.R
Rscript overfitting.R
Rscript boosting.R
Rscript ensemble_methods.R
```

## Code Examples Overview

### 1. Decision Trees Introduction (`decision_trees_introduction.py/.R`)

**Key Concepts Demonstrated:**
- Recursive splitting and region partitioning
- Entropy loss and Gini impurity calculations
- Information gain and greedy splitting
- Regression trees with least-squares loss
- Regularization techniques (max_depth, min_samples_leaf, etc.)
- Runtime complexity analysis
- Decision boundary visualization

**Main Functions:**
- `demonstrate_recursive_splitting()` - Shows how trees split regions
- `compare_entropy_gini()` - Compares different impurity measures
- `regression_tree_demo()` - Demonstrates regression trees
- `regularization_demo()` - Shows regularization effects
- `runtime_complexity_demo()` - Analyzes computational complexity
- `visualize_decision_boundaries()` - Creates decision boundary plots

### 2. Loan Example (`loan_example.py/.R`)

**Key Concepts Demonstrated:**
- Real-world decision tree application
- Feature engineering for categorical variables
- Tree construction and evaluation
- Decision path analysis
- Loan scoring and prediction
- Feature importance analysis
- Model interpretation

**Main Functions:**
- `create_loan_dataset()` - Generates synthetic loan data
- `preprocess_data()` - Handles categorical variables
- `train_tree()` - Trains decision tree classifier
- `evaluate_tree()` - Evaluates model performance
- `demonstrate_loan_scoring()` - Shows loan application scoring
- `analyze_decision_paths()` - Analyzes decision paths
- `create_loan_visualizations()` - Creates data analysis plots

### 3. Overfitting (`overfitting.py/.R`)

**Key Concepts Demonstrated:**
- Overfitting in decision trees
- Early stopping techniques
- Pruning methods (cost complexity pruning)
- Regularization parameter comparison
- Bias-variance trade-off analysis
- Cross-validation for model selection

**Main Functions:**
- `demonstrate_depth_vs_performance()` - Shows depth vs performance relationship
- `early_stopping_demo()` - Demonstrates early stopping
- `pruning_demo()` - Shows pruning techniques
- `regularization_comparison()` - Compares regularization methods
- `bias_variance_analysis()` - Analyzes bias-variance trade-off
- `visualize_overfitting()` - Visualizes overfitting effects

### 4. Boosting (`boosting.py/.R`)

**Key Concepts Demonstrated:**
- Weak learners and their limitations
- AdaBoost algorithm
- Estimator weights and progression
- Boosting vs single deep trees
- Robustness to noise
- Cross-validation for boosting

**Main Functions:**
- `create_weak_learners_demo()` - Shows weak learner limitations
- `adaboost_demo()` - Demonstrates AdaBoost algorithm
- `visualize_adaboost_progression()` - Shows boosting progression
- `analyze_estimator_weights()` - Analyzes estimator weights
- `compare_with_single_tree()` - Compares with deep trees
- `demonstrate_boosting_robustness()` - Shows robustness to noise

### 5. Ensemble Methods (`ensemble_methods.py/.R`)

**Key Concepts Demonstrated:**
- Single tree limitations
- Bagging (Bootstrap Aggregating)
- Random Forests
- Feature importance analysis
- Ensemble method comparison
- Robustness analysis

**Main Functions:**
- `demonstrate_single_tree_limitations()` - Shows single tree problems
- `bagging_demo()` - Demonstrates bagging
- `random_forest_demo()` - Shows Random Forests
- `analyze_ensemble_performance()` - Compares ensemble methods
- `analyze_feature_importance()` - Analyzes feature importance
- `compare_ensemble_methods()` - Compares all methods
- `demonstrate_ensemble_robustness()` - Shows robustness

## Output Files

Each example generates several output files:

### Visualizations
- Decision boundary plots
- Performance comparison charts
- Feature importance plots
- Overfitting analysis plots
- Ensemble method comparisons

### Data Files
- Model performance metrics
- Cross-validation results
- Feature importance rankings
- Comparison tables

## Key Features

### Comprehensive Coverage
- All major decision tree concepts
- Both classification and regression
- Multiple ensemble methods
- Real-world applications

### Educational Focus
- Step-by-step demonstrations
- Clear explanations in comments
- Visual outputs for understanding
- Performance comparisons

### Practical Implementation
- Production-ready code structure
- Error handling and validation
- Reproducible results (fixed random seeds)
- Modular design for easy modification

### Cross-Language Support
- Identical functionality in Python and R
- Same output formats and visualizations
- Consistent naming conventions
- Parallel structure for easy comparison

## Customization

All examples are designed to be easily customizable:

- **Dataset parameters**: Modify noise levels, sample sizes, feature counts
- **Model parameters**: Adjust tree depth, regularization, ensemble size
- **Visualization**: Change plot styles, colors, layouts
- **Analysis**: Add new metrics, comparison methods, evaluation criteria

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using pip/R install.packages()
2. **Memory Issues**: Reduce dataset sizes or ensemble sizes
3. **Plot Display**: Ensure matplotlib backend is properly configured
4. **R Package Conflicts**: Check for package version compatibility

### Performance Tips

- Use smaller datasets for quick testing
- Reduce ensemble sizes for faster execution
- Disable visualization for batch processing
- Use parallel processing for large ensembles

## Contributing

To add new examples or improve existing ones:

1. Follow the established code structure
2. Include comprehensive documentation
3. Add both Python and R versions
4. Ensure reproducible results
5. Include appropriate visualizations

## References

- Elements of Statistical Learning (ESL)
- Introduction to Statistical Learning (ISL)
- Scikit-learn documentation
- R documentation for tree and ensemble packages
