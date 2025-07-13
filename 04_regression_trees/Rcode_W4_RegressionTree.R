# (PSL) Regression Tree

# Load required libraries
library(ggplot2)
library(rpart)
library(rpart.plot)
library(tree)

# Load the housing dataset
url = "https://liangfgithub.github.io/Data/HousingData.csv"
Housing = read.csv(url)

# There are two R packages for tree models, 'tree' and 'rpart'. 
# We will mainly use 'rpart'. The package 'tree' is called for its 
# command 'partition.tree', which we use to generate the first figure.

# The tree Package
# The syntax for fitting a regression tree with 'tree' is similar to that 
# used for linear regression models. In this example, we use just two 
# predictors, longitude and latitude, from the Housing data to predict 
# the variable Y.

trfit = tree(Y ~ lon + lat, data = Housing)
small.tree = prune.tree(trfit, best = 7)
small.tree

# Plot the tree
par(mfrow = c(1, 2))
plot(small.tree)
text(small.tree, cex = .75)

# Create price quantiles for visualization
price.quantiles = cut(Housing$Y, quantile(Housing$Y, 0:20 / 20),
  include.lowest = TRUE)
plot(Housing$lat, Housing$lon, col = grey(20:1 / 21)[price.quantiles],
  pch = 20, ylab = "Longitude", xlab = "Latitude")
partition.tree(small.tree, ordvars = c("lat", "lon"), add = TRUE)

# Detach the 'tree' package and switch to the 'rpart' package.
detach("package:tree")

# The rpart Package

# Fit a Tree
# The syntax for fitting a regression tree with 'rpart' closely resembles 
# that used for linear regression models. In this example, we use the 
# housing dataset to build a regression tree that predicts the target 
# variable Y, using all other available columns as potential predictors. 
# The default tree plot may not be visually appealing, so for better 
# visualization, we can use the 'rpart.plot' package.

set.seed(1234)
tr1 = rpart(Y~., data = Housing)
par(mfrow = c(1, 2))
plot(tr1)
rpart.plot(tr1)

# Understanding the Fitted Tree:
# Starting at the root node, the average value for all observations is 3, 
# representing 100% of the data points. The first split occurs based on 
# the variable 'lstat'. The left node contains roughly 35% of the samples 
# with an average of 2.7, while the right node contains the remaining 65% 
# with an average of 3.2. Leaf nodes appear at the bottom, and their color 
# shading indicates the magnitude of their predictions.

# Cross-validation
# Cross-validation is automatically performed when using the 'rpart' command. 
# To inspect the cross-validation for pruning, use the 'printcp' command, 
# which provides a table.

# - 'CP' means **Complexity Parameter**, which is a scaled version of α 
#   (the price or penalty we pay for keeping a split)
# - 'nsplit' means the number of splits; the number of leaf nodes 
#   (or size of a tree) is equal to 'nsplit' + 1 
# - 'rel error' and 'xerror' are the training error (RSS) and CV error, 
#   but these values are relative to a reference training error at the root node.

# Connection between α and CP
# RSS(T) + α |T|, \quad \frac{RSS(T)}{RSS(\text{root})} + CP \cdot |T|

printcp(tr1)

# Pruning
# We can employ the 'prune' command to obtain the optimal subtree by 
# providing specific Complexity Parameter (CP) values.

prune(tr1, cp = 0.3)
prune(tr1, cp = 0.2)
prune(tr1, cp = 0.156)

# A CV path plot
plotcp(tr1)

# If you're uncertain whether the cross-validation error has reached
# its minimum, consider starting with a larger tree.

tr2 = rpart(Y~., data = Housing,
  control = list(cp = 0, xval = 10))
printcp(tr2)
plot(tr2)
plotcp(tr2)

# Optimal CP Value
# We can use either the minimum cross-validation error or the One
# Standard Error (1-SE) rule to choose the optimal CP value for pruning.

# index of CP with lowest xerror
opt = which.min(tr2$cptable[, "xerror"]) 
# the optimal CP value
tr2$cptable[opt, 4]

# upper bound for equivalent optimal xerror
tr2$cptable[opt, 4] + tr2$cptable[opt, 5]

# row IDs for CPs whose xerror are equivalent to min(xerror)
tmp.id = which(tr2$cptable[, 4] <= tr2$cptable[opt, 4] +
  tr2$cptable[opt, 5])
tmp.id = min(tmp.id)
# CP.1se = any value between row(tmp.id) and(tmp.id - 1)
CP.1se = (tr2$cptable[tmp.id -1, 1] + tr2$cptable[tmp.id, 1]) / 2

# Prune tree with CP.1se
tr3 = prune(tr2, cp = CP.1se)

# More on the CP Table
# Understand the relationship between the 1st column and the 3rd column
# of the CP table.

# The difference between adjacent 'rel error' in the table
# is equivalent to the corresponding CP values. This difference signifies
# the improvement gained in RSS by adding an additional split, effectively
# setting the "cost" of that split.

cbind(tr2$cptable[, 1], c(-diff(tr2$cptable[, 3]), 0))

# Making Predictions
# After the tree is fit, the 'predict' function from the
# 'rpart' package can be used for making predictions on new or
# existing data.

?predict.rpart 