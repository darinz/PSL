# Statistical Learning

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

### Upcoming Modules
- **Regularization Methods**: Ridge, Lasso, Elastic Net
- **Classification Methods**: Logistic regression, LDA, QDA
- **Resampling Methods**: Cross-validation, bootstrap
- **Model Selection**: Information criteria and validation strategies
- **Tree-Based Methods**: Decision trees, bagging, random forests
- **Support Vector Machines**: Linear and non-linear classification
- **Unsupervised Learning**: Clustering and dimensionality reduction

## Module Structure

### 📁 01_introduction/
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

### 📁 02_linear_regression/
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

### 📁 reference/
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
pip3 install numpy pandas matplotlib seaborn scikit-learn jupyter statsmodels
```

#### 5. Install Required R Packages

Open R or RStudio and install the necessary packages:

```r
# Core statistical packages
install.packages(c("ggplot2", "dplyr", "tidyr", "caret", "randomForest"))

# Additional packages for simulations and analysis
install.packages(c("MASS", "mvtnorm", "class", "e1071", "car"))
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

3. **For interactive exploration, use Jupyter:**
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
   - Use the PDF references in `reference/` for additional reading

4. **Study the materials systematically**
   - Read the markdown notes for theoretical understanding
   - Run the code examples to reinforce concepts
   - Review visualizations in `img/` directories for intuitive understanding

5. **Practice and experiment**
   - Modify parameters in simulation scripts
   - Apply concepts to your own datasets
   - Compare Python and R implementations

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

