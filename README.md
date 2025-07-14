# Statistical Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive resource for learning Practical Statistical Learning, bridging the gap between statistical theory and modern machine learning practice. This repository provides structured notes, visual aids, and reference materials to support both self-study and formal coursework.

## Table of Contents

- [Introduction](#introduction)
- [Curriculum Topics](#curriculum-topics)
- [Module Structure](#module-structure)
- [Development Environment Setup](#development-environment-setup)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This repository is designed to facilitate a deep understanding of statistical learning, with a focus on both theoretical foundations and practical applications. It includes:

- Well-organized markdown notes summarizing key concepts and mathematical derivations
- Comprehensive code examples in both Python and R
- High-quality images and diagrams to illustrate statistical learning concepts
- PDF reference materials for in-depth reading
- Structured learning modules with clear progression paths

## Complementary Learning Resources

For a comprehensive introduction to machine learning that complements this statistical learning material, see:

- **[Machine Learning Repository](https://github.com/darinz/Machine-Learning)**: A comprehensive resource covering supervised and unsupervised learning, learning theory, reinforcement learning, and modern applications including deep learning, transformers, and foundation models. This repository provides:
  - Linear models and generative learning algorithms
  - Advanced classification techniques
  - Deep learning fundamentals and neural networks
  - Clustering and EM algorithms with variational methods
  - Dimensionality reduction techniques (PCA, ICA)
  - Self-supervised learning and foundation models
  - Reinforcement learning and control systems
  - Modern applications in NLP, computer vision, and robotics

The Machine Learning repository offers a broader perspective on algorithmic approaches and modern machine learning techniques, while this Statistical Learning repository focuses on the statistical foundations and classical methods. Together, they provide a complete learning path from statistical theory to modern machine learning practice.

## Curriculum Topics

### Module 1: Introduction to Statistical Learning
- **Types of Statistical Learning Problems**: Supervised vs. Unsupervised Learning
- **Learning Theory**: Mathematical foundations and statistical decision theory
- **Bias-Variance Tradeoff**: The fundamental tension in model complexity
- **Fundamental Algorithms**: Linear regression, k-Nearest Neighbors, Bayes classifier
- **Model Selection**: Cross-validation and regularization techniques

### Module 2: Linear Regression
- **Multiple Linear Regression**: Matrix formulation and least squares estimation
- **Geometric Interpretation**: Vector spaces, projections, and orthogonality
- **Practical Implementation**: Data analysis, diagnostics, and model validation
- **Advanced Concepts**: Partial effects, hypothesis testing, and model comparison

### Module 3: Variable Selection and Regularization
- **Subset Selection**: Best subset, forward/backward stepwise selection
- **Regularization Methods**: Ridge regression, Lasso, Elastic Net
- **Model Selection**: Cross-validation, information criteria
- **Practical Considerations**: Multicollinearity, variable importance

### Module 4: Regression Trees and Ensemble Methods
- **Decision Trees**: CART algorithm, tree construction and pruning
- **Random Forests**: Bagging, feature importance, out-of-bag estimation
- **Gradient Boosting**: Boosting algorithms, parameter tuning
- **Ensemble Methods**: Combining multiple models for improved performance

### Module 5: Nonlinear Regression
- **Polynomial Regression**: Higher-order terms and polynomial basis functions
- **Cubic Splines**: Basis functions and knot placement
- **Regression Splines**: Natural splines and smoothing splines
- **Local Regression**: LOESS and kernel smoothing methods

### Module 6: Clustering Analysis (Coming Soon)
- **K-Means Clustering**: Partitioning methods and centroid-based clustering
- **Hierarchical Clustering**: Agglomerative and divisive methods
- **Density-Based Clustering**: DBSCAN and density estimation
- **Model-Based Clustering**: Gaussian mixture models and expectation-maximization

## Module Structure

### 01_introduction/
Comprehensive introduction to statistical learning fundamentals:

**Core Theory:**
- `01_introduction.md` - Problem types and learning paradigms
- `02_learning_theory.md` - Mathematical foundations and decision theory
- `03_bias_variance.md` - Deep dive into the bias-variance tradeoff
- `04_ls_and_knn.md` - Linear regression and k-Nearest Neighbors
- `05_bayes_rule.md` - Bayes classification rule and optimality

**Practical Implementation:**
- `Python_W1_SimulationStudy.py` - Python simulation studies
- `Rcode_W1_SimulationStudy.R` - R simulation studies
- `img/` - Supporting visualizations and diagrams

### 02_linear_regression/
Advanced treatment of linear regression methods:

**Theoretical Foundations:**
- `01_mulitple_linear_regression.md` - Matrix formulation and estimation
- `02_geometric_interpretation.md` - Vector space interpretation
- `03_practical_issues.md` - Implementation and diagnostics

**Code Examples:**
- `Python_W2_LinearRegression_1.py` - Basic regression analysis
- `Python_W2_LinearRegression_2.py` - Advanced concepts and diagnostics
- `Rcode_W2_LinearRegression.R` - Comprehensive R implementation
- `img/` - Geometric and diagnostic visualizations

### 03_variable_selection_regularization/
Variable selection and regularization techniques:

**Theoretical Foundations:**
- `01_subset_selection.md` - Best subset, forward/backward stepwise selection
- `02_regularization.md` - Regularization principles and methods
- `03_ridge_regression.md` - Ridge regression theory and implementation
- `04_lasso_regression.md` - Lasso regression and variable selection
- `05_discussion.md` - Comparison and practical considerations

**Code Examples:**
- `Python_W3_VarSel_SubsetSelection.py` - Subset selection in Python
- `Python_W3_VarSel_RidgeLasso.py` - Ridge and Lasso implementation
- `R_W3_VarSel_SubsetSelection.R` - Subset selection in R
- `Rcode_W3_VarSel_RidgeLasso.R` - Ridge and Lasso in R

### 04_regression_trees/
Tree-based methods and ensemble learning:

**Theoretical Foundations:**
- `01_regression_trees.md` - Decision tree construction and CART algorithm
- `02_random_forest.md` - Random forests and bagging methods
- `03_gbm.md` - Gradient boosting machines and boosting algorithms

**Code Examples:**
- `Python_W4_RegressionTree.py` - Decision tree implementation
- `Python_W4_Regression_RandomForest.py` - Random forest in Python
- `Python_W4_Regression_GBM.py` - Gradient boosting in Python
- `Rcode_W4_RegressionTree.R` - Decision tree in R
- `Rcode_W4_Regression_RandomForest.R` - Random forest in R
- `Rcode_W4_Regression_GBM.R` - Gradient boosting in R

### 05_nonlinear_regression/
Nonlinear regression and smoothing methods:

**Theoretical Foundations:**
- `01_polynomial_regression.md` - Polynomial basis functions and higher-order terms
- `02_cubic_splines.md` - Cubic spline basis functions and knot placement
- `03_regression_splines.md` - Natural splines and regression splines
- `04_smoothing_splines.md` - Smoothing splines and regularization
- `05_local_regression.md` - Local regression and kernel smoothing

**Code Examples:**
- `Python_W5_PolynomialRegression.py` - Polynomial regression implementation
- `Python_W5_RegressionSpline.py` - Regression splines in Python
- `Python_W5_SmoothingSpline.html` - Interactive smoothing spline examples
- `Python_W5_LocalSmoother.html` - Interactive local regression examples
- `Rcode_W5_PolynomialRegression.R` - Polynomial regression in R
- `Rcode_W5_RegressionSpline.R` - Regression splines in R
- `Rcode_W5_SmoothingSpline.html` - Interactive R smoothing spline examples
- `Rcode_W5_LocalSmoother.html` - Interactive R local regression examples

### 06_clustering_analysis/
Clustering and unsupervised learning methods (Coming Soon)

### _images/
Global image repository containing visualizations from all modules:
- Week 3: Variable selection and regularization diagrams
- Week 4: Tree-based methods and ensemble learning visualizations
- Week 5: Nonlinear regression and smoothing spline illustrations

### reference/
Essential reference materials:
- `ESLII.pdf` - The Elements of Statistical Learning (2nd Edition)
- `ISLRv2.pdf` - An Introduction to Statistical Learning (2nd Edition)
- `ISLP.pdf` - Introduction to Statistical Learning with Python
- `R_for_Statistical_Learning.pdf` - R-specific learning materials

## Development Environment Setup

### Prerequisites

Before running the example code, ensure you have the following installed:

- **Git**: For cloning the repository
- **Python 3.8+**: For Python simulations and analysis
- **R 4.0+**: For R simulations and statistical analysis
- **RStudio**: Recommended IDE for R development

### Installation Instructions

#### 1. Install Python

**macOS (using Homebrew):**
```bash
brew install python
```

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Ensure "Add Python to PATH" is checked during installation

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**Verify installation:**
```bash
python3 --version
pip3 --version
```

#### 2. Install R

**macOS:**
- Download from [r-project.org](https://cran.r-project.org/bin/macosx/)
- Or use Homebrew: `brew install r`

**Windows:**
- Download from [r-project.org](https://cran.r-project.org/bin/windows/base/)

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install r-base r-base-dev
```

**Verify installation:**
```bash
R --version
```

#### 3. Install RStudio

**All platforms:**
- Download from [rstudio.com](https://posit.co/download/rstudio-desktop/)
- Follow the installation wizard

**Alternative for Linux:**
```bash
sudo apt install rstudio
```

#### 4. Install Required Python Packages

Navigate to the repository directory and install dependencies:

```bash
cd Statistical-Learning
pip3 install numpy pandas matplotlib seaborn scikit-learn jupyter statsmodels plotly
```

#### 5. Install Required R Packages

Open R or RStudio and install the necessary packages:

```r
# Core statistical packages
install.packages(c("ggplot2", "dplyr", "tidyr", "caret", "randomForest"))

# Additional packages for simulations and analysis
install.packages(c("MASS", "mvtnorm", "class", "e1071", "car", "gbm", "splines"))
```

### Running Example Code

#### Python Simulations

1. **Introduction Module:**
   ```bash
   cd 01_introduction
   python3 Python_W1_SimulationStudy.py
   ```

2. **Linear Regression Module:**
   ```bash
   cd 02_linear_regression
   python3 Python_W2_LinearRegression_1.py
   python3 Python_W2_LinearRegression_2.py
   ```

3. **Variable Selection and Regularization:**
   ```bash
   cd 03_variable_selection_regularization
   python3 Python_W3_VarSel_SubsetSelection.py
   python3 Python_W3_VarSel_RidgeLasso.py
   ```

4. **Regression Trees and Ensemble Methods:**
   ```bash
   cd 04_regression_trees
   python3 Python_W4_RegressionTree.py
   python3 Python_W4_Regression_RandomForest.py
   python3 Python_W4_Regression_GBM.py
   ```

5. **Nonlinear Regression:**
   ```bash
   cd 05_nonlinear_regression
   python3 Python_W5_PolynomialRegression.py
   python3 Python_W5_RegressionSpline.py
   ```

6. **For interactive exploration, use Jupyter:**
   ```bash
   jupyter notebook
   ```

#### R Simulations

1. **Using RStudio:**
   - Open RStudio
   - Open the respective R files in each module
   - Run the entire script or execute sections individually

2. **Using R console:**
   ```bash
   cd 01_introduction
   Rscript Rcode_W1_SimulationStudy.R
   
   cd ../02_linear_regression
   Rscript Rcode_W2_LinearRegression.R
   
   cd ../03_variable_selection_regularization
   Rscript Rcode_W3_VarSel_RidgeLasso.R
   
   cd ../04_regression_trees
   Rscript Rcode_W4_RegressionTree.R
   Rscript Rcode_W4_Regression_RandomForest.R
   Rscript Rcode_W4_Regression_GBM.R
   
   cd ../05_nonlinear_regression
   Rscript Rcode_W5_PolynomialRegression.R
   Rscript Rcode_W5_RegressionSpline.R
   ```

3. **For interactive R sessions:**
   ```bash
   R
   source("Rcode_W1_SimulationStudy.R")
   ```

### Expected Outputs

The simulation scripts will generate:
- **Plots**: Bias-variance tradeoff visualizations, model complexity curves, diagnostic plots
- **Statistical summaries**: Performance metrics, coefficient estimates, hypothesis tests
- **Console output**: Analysis results, model diagnostics, and parameter estimates
- **Interactive visualizations**: HTML files for exploring nonlinear regression methods

### Troubleshooting

**Python Issues:**
- Ensure all required packages are installed: `pip3 list`
- Check Python version compatibility: `python3 --version`
- For plotting issues, ensure a display environment is available

**R Issues:**
- Verify package installation: `installed.packages()`
- Check R version: `R.version.string`
- For plotting in headless environments, use: `pdf("output.pdf")` before plots

**General Issues:**
- Ensure you're in the correct directory when running scripts
- Check file permissions for execution
- Verify all dependencies are properly installed

## How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/darinz/Statistical-Learning.git
   cd Statistical-Learning
   ```

2. **Set up your development environment** (see instructions above)

3. **Follow the structured learning path**
   - Start with `01_introduction/` for foundational concepts
   - Progress to `02_linear_regression/` for advanced regression methods
   - Continue with `03_variable_selection_regularization/` for regularization techniques
   - Explore `04_regression_trees/` for tree-based methods
   - Study `05_nonlinear_regression/` for nonlinear modeling approaches
   - Use the PDF references in `reference/` for additional reading

4. **Study the materials systematically**
   - Read the markdown notes for theoretical understanding
   - Run the code examples to reinforce concepts
   - Review visualizations in `img/` directories and `_images/` for intuitive understanding

5. **Practice and experiment**
   - Modify parameters in simulation scripts
   - Apply concepts to your own datasets
   - Compare Python and R implementations
   - Explore interactive visualizations for nonlinear methods

## Contributing

Contributions are welcome to improve the clarity, accuracy, and breadth of this resource. You can:

- Report issues or suggest improvements via the [Issues](https://github.com/darinz/Statistical-Learning/issues) page
- Submit pull requests for corrections, new notes, or additional resources
- Help enhance documentation and add new visualizations
- Contribute new modules or expand existing ones

### Guidelines

- Fork the repository and create a feature branch
- Make your changes with clear, descriptive commit messages
- Open a pull request with a summary of your contribution
- Ensure code examples run successfully
- Update documentation to reflect any changes

## Acknowledgements

This repository draws heavily from the following primary resources:

### Primary Resources
- **[PSL Online Notes](https://liangfgithub.github.io/PSL/)**: Main course reference
- **[An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/index.html)**: Beginner-friendly textbook
- **[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)**: Advanced reference text

These materials provide the theoretical foundation and practical examples that form the basis of the notes and explanations contained in this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

