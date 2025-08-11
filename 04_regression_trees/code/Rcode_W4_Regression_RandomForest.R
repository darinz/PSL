# (PSL) Random Forest for Regression
# Let’s discuss how to fit a Random Forest model in R. To begin, we
# load the randomForest package—note that the ‘F’ is uppercase. We then load the housing dataset and split it into a training
# set (70%) and a test set (30%).
library(randomForest)
# randomForest 4.7-1.1
# Type rfNews() to see new features/changes/bug fixes.

url = "https://liangfgithub.github.io/Data/HousingData.csv"
mydata = read.csv(url)
n = nrow(mydata)
ntest = round(n * 0.3)
set.seed(134)
test.id = sample(1:n, ntest)

# Model Fitting
# To fit the model, we use the randomForest function,
# employing syntax similar to standard predictive modeling functions in R.
# We aim to predict the variable Y using all other columns in the data
# matrix as potential predictors. We set the importance parameter to TRUE
# and specify that the forest should consist of 400 trees.
names(mydata)
#  [1] "Y"       "chas"    "lon"     "lat"     "crim"    "zn"      "indus"  
#  [8] "nox"     "rm"      "age"     "dis"     "rad"     "tax"     "ptratio"
# [15] "lstat"

rfModel = randomForest(Y ~ ., data = mydata[-test.id, ],
                       importance = T, ntree=400); 

# Explaining Outputs
# After fitting the model, various outputs are generated.
names(rfModel)
# One key parameter is mtry, which indicates the number of
# variables selected for each split. In our example, mtry is
# set to 4, aligning with the recommended value for regression-based
# Random Forest models: p/3, where p is the number of predictors.
# the default value for mtry is p/3 for regression
# p = ncol(mydata) - 1 = 14
# mtry = 14/3 = 4
rfModel$mtry

# Making Predictions
# To generate predictions, we use the fitted model and provide the
# feature matrix (X) for the test set. We can then calculate the average
# test error, which, in our case, is approximately 3%.
yhat.test = predict(rfModel, mydata[test.id, ])
sum((mydata$Y[test.id] - yhat.test)^2)/length(test.id)
# [1] 0.03093106

# Understanding Training Errors
# It’s worth discussing the training error. Random Forest provides
# two types of training errors.
# The first is obtained by calling the prediction function on the
# training data, resulting in an error rate, 0.45%, substantially lower
# than the test error. This is not surprising, as training errors are
# generally lower than test errors, especially for complex models.
yhat.train = predict(rfModel, mydata[-test.id, ])
sum((mydata$Y[-test.id] - yhat.train) ^2)/(n - ntest)
# [1] 0.004530652
# However, Random Forest also provides a second type of training
# error—based on so-called out-of-bag samples. This error
# rate tends to be closer to the test error and serves as a more reliable
# estimator of the model’s generalization performance.
sum((mydata$Y[-test.id] - rfModel$predicted) ^2)/(n - ntest)
# [1] 0.02201958

# Out-of-Bag Samples
# In the Random Forest procedure, each tree is built using one such
# bootstrap sample. The data points not included in a specific bag are termed “Out-of-Bag” (OOB) samples for that tree. These samples serve as test points for evaluating that tree’s performance.
# In Random Forest’s output, there’s a component called oob.times that indicates, for each training instance, how many times it was an OOB sample.
rfModel$oob.times[1:5]
# [1] 148 152 134 150 141
# length(rfModel$oob)
# To delve a bit deeper, the average number of times a sample becomes
# an OOB can be calculated. This figure is derived from the mathematical
# limit that arises when considering the probability of a single instance
# not being included in a bootstrap sample. The formula (1 - 1/n )^n approximates
# exp(-1) as n becomes large. In our case, multiplying this limit by the
# total number of trees (400) gives us an average Out-of-Bag count of 147,
# corroborating our observations.
# oob.times --> ntree * exp(-1) = ntree * 0.368
rfModel$ntree * exp(-1)
# [1] 147.1518
mean(rfModel$oob.times)
# [1] 146.5791

# Performance Plots
# Next, let’s visually examine the performance of our Random Forest
# model using two plots.
# The first is generated using the Random Forest’s built-in plotting
# function. The second is our custom reproduction of the first, created to
# ensure we fully understand what exactly is depicted in the default
# plot.
# Both plots feature the number of trees on the x-axis and their
# corresponding mean square error (MSE) on the y-axis; each MSE value on
# the y-axis is calculated based on a forest with the specific number of
# trees marked on the x-axis. These MSE values are derived from OOB
# predictions, making them a reliable estimate more akin to
# cross-validation error than to simple training error.
# The plots reveal that the model’s performance improves rapidly with
# the initial addition of trees but starts to plateau beyond a certain
# point. This suggests that adding more trees does not lead to
# overfitting, although one could potentially achieve similar performance
# with fewer trees, say, just the first 100.
par(mfrow=c(1, 2))
plot(rfModel)

tmp = rfModel$mse
plot(c(0, rfModel$ntree), range(tmp), type="n",
     xlab = "Number of trees", ylab="Error")
lines(tmp) 