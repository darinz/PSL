# Statistical Learning


A comprehensive resource for learning Practical Statistical Learning, bridging the gap between statistical theory and modern machine learning practice. This repository provides structured notes, visual aids, and reference materials to support both self-study and formal coursework.

## Table of Contents

- [Introduction](#introduction)
- [Curriculum Topics](#curriculum-topics)
- [Development Environment Setup](#development-environment-setup)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This repository is designed to facilitate a deep understanding of statistical learning, with a focus on both theoretical foundations and practical applications. It includes:

- Well-organized markdown notes summarizing key concepts and mathematical derivations
- HTML reference materials for in-depth reading
- High-quality images and diagrams to illustrate statistical learning problems, bias-variance tradeoff, and more

## Curriculum Topics

- Introduction to Statistical Learning
- Types of Statistical Learning Problems
- Supervised vs. Unsupervised Learning
- Curse of Dimensionality
- Bias-Variance Tradeoff
- Model Complexity and Regularization
- Modern Topics: Double Descent, Ensemble Methods, and more

## Development Environment Setup

### Prerequisites

Before running the example code, ensure you have the following installed:

- **Git**: For cloning the repository
- **Python 3.8+**: For Python simulations
- **R 4.0+**: For R simulations
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
pip3 install numpy pandas matplotlib seaborn scikit-learn jupyter
```

#### 5. Install Required R Packages

Open R or RStudio and install the necessary packages:

```r
# Core statistical packages
install.packages(c("ggplot2", "dplyr", "tidyr", "caret", "randomForest"))

# Additional packages for simulations
install.packages(c("MASS", "mvtnorm", "class", "e1071"))
```

### Running Example Code

#### Python Simulations

1. **Navigate to the introduction module:**
   ```bash
   cd 01_introduction
   ```

2. **Run the Python simulation study:**
   ```bash
   python3 Python_W1_SimulationStudy.py
   ```

3. **For interactive exploration, use Jupyter:**
   ```bash
   jupyter notebook
   ```
   Then open and run the Python simulation file in the notebook interface.

#### R Simulations

1. **Using RStudio:**
   - Open RStudio
   - Open the `Rcode_W1_SimulationStudy.R` file
   - Run the entire script or execute sections individually

2. **Using R console:**
   ```bash
   cd 01_introduction
   Rscript Rcode_W1_SimulationStudy.R
   ```

3. **For interactive R sessions:**
   ```bash
   R
   source("Rcode_W1_SimulationStudy.R")
   ```

### Expected Outputs

The simulation scripts will generate:
- **Plots**: Bias-variance tradeoff visualizations, model complexity curves
- **Statistical summaries**: Performance metrics and comparisons
- **Console output**: Analysis results and parameter estimates

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

3. **Browse the notes**
   - Start with `01_introduction/01_introduction.md` for foundational concepts
   - Explore topic-specific markdown files and HTML references for details
   - Review images in `img/` for visual explanations

4. **Run example code**
   - Execute Python simulations for bias-variance analysis
   - Run R simulations for statistical modeling
   - Experiment with parameters to observe effects

5. **Supplement your study**
   - Use the PDF in `reference/` for additional reading
   - Integrate these notes with your curriculum work or self-study

## Contributing

Contributions are welcome to improve the clarity, accuracy, and breadth of this resource. You can:

- Report issues or suggest improvements via the [Issues](https://github.com/darinz/Statistical-Learning/issues) page
- Submit pull requests for corrections, new notes, or additional resources
- Help enhance documentation and add new visualizations

### Guidelines

- Fork the repository and create a feature branch
- Make your changes with clear, descriptive commit messages
- Open a pull request with a summary of your contribution

## Acknowledgements

This repository draws heavily from the following primary resources:

### Primary Resources
- **[PSL Online Notes](https://liangfgithub.github.io/PSL/)**: Main course reference
- **[An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/index.html)**: Beginner-friendly textbook
- **[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)**: Advanced reference text

These materials provide the theoretical foundation and practical examples that form the basis of the notes and explanations contained in this repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

