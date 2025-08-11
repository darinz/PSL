# (PSL) GBM for Regression

# Preliminaries
# Before running a GBM model, decide on the following hyperparameters:
# - loss Function: For regression tasks, ‘Gaussian’ (squared error) is commonly used
# - n.trees: number of trees (default = 100)
# - shrinkage: shrinkage factor or learning rate (default = 0.1)
# - bag.fraction: fraction of the training data used for learning (default = 0.5)
# - cv.folds: number of folds for cross-validation (default = 0, i.e., no CV error returned)
# - interaction.depth: depth of individual trees (default 1)

# Case Study: Housing Data
# We first split the Boston Housing dataset into a 70% training set and a 30% test set.
library(gbm)

## Loaded gbm 2.1.8.1

url = "https://liangfgithub.github.io/Data/HousingData.csv"
mydata = read.csv(url)
n = nrow(mydata)
ntest = round(n * 0.3)
set.seed(1234)
test.id = sample(1:n, ntest)

# Fit a GBM
myfit1 = gbm(Y ~ . , data = mydata[-test.id, ],
            distribution = "gaussian",
            n.trees = 100,
            shrinkage = 1,
            interaction.depth = 3,
            bag.fraction = 1,
            cv.folds = 5)
myfit1

## gbm(formula = Y ~ ., distribution = "gaussian", data = mydata[-test.id, 
##     ], n.trees = 100, interaction.depth = 3, shrinkage = 1, bag.fraction = 1, 
##     cv.folds = 5)
## A gradient boosted model with gaussian loss function.
## 100 iterations were performed.
## The best cross-validation iteration was 20.
## There were 14 predictors of which 13 had non-zero influence.

# Optimal Stopping Point
# Plot the CV error to find the optimal number of trees to prevent overfitting.
opt.size = gbm.perf(myfit1) 