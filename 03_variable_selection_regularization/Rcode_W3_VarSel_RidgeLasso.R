# Stat542: Ridge and Lasso Regression
# Fall 2022
# R code extracted from HTML notebook

# Load required library
library(glmnet)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

# Load the Prostate data
myData = read.table(file = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data", 
                    header = TRUE)
myData = myData[, -10]  # Remove the 10th column
names(myData)[9] = 'Y'  # Rename response column
names(myData)

# Data dimensions
n = dim(myData)[1]          # sample size
p = dim(myData)[2] - 1      # number of non-intercept predictors
X = as.matrix(myData[, names(myData) != "Y"]);   # some algorithms need the matrix/vector 
Y = myData$Y;       

# Train/Test Split
ntest = round(n*0.2)
ntrain = n - ntest;
test.id = sample(1:n, ntest)
Ytest = Y[test.id]

# ============================================================================
# FULL LINEAR MODEL (BASELINE)
# ============================================================================

# Fit ordinary linear regression using all variables
full.model = lm( Y ~ ., data = myData[-test.id, ]);  
Ytest.pred = predict(full.model, newdata= myData[test.id, ])
sum((Ytest - Ytest.pred)^2)/ntest # averaged MSE on the test set

# ============================================================================
# RIDGE REGRESSION
# ============================================================================

# Fit ridge regression models using glmnet with default lambda sequence
myridge = glmnet(X[-test.id, ], Y[-test.id], alpha = 0)
plot(myridge, label = TRUE, xvar = "lambda")

# Check output from glmnet for ridge regression
summary(myridge)

# Retrieve lambda values
length(myridge$lambda)

# Retrieve coefficients using two different approaches
dim(myridge$beta)        # coefficients for non-intercept predictors
length(myridge$a0)       # intercept

# Retrieve coefficients (including intercept) using coef(myridge)
dim(coef(myridge))

# The two coefficient matrices should be the same
sum((coef(myridge) - rbind(myridge$a0, myridge$beta))^2)

# Note: Ridge regression coefficients may change signs along the path
round(myridge$beta['age', ], dig = 4)

# How are the intercepts computed?
k = 2; 
my.mean = apply(X[-test.id, ], 2, mean)  # 1x8 mean vector for training X
mean(Y[-test.id]) - sum(my.mean * myridge$beta[, k])
myridge$a0[k] # intercept for lambda = myridge$lambda[k]

# Check whether our intercept formula is true for all intercepts 
sum((mean(Y[-test.id]) - my.mean %*% myridge$beta  - myridge$a0)^2)

# ============================================================================
# RIDGE REGRESSION - CROSS VALIDATION
# ============================================================================

# Cross-validation for ridge regression
cv.out = cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 0) 
plot(cv.out)

# The CV performance continues to drop when lambda gets smaller
# So we provide R with a lambda sequence dense in [-6, 2]
lam.seq = exp(seq(-6, 2, length=100))
cv.out = cv.glmnet(X[-test.id,], Y[-test.id], alpha=0, lambda=lam.seq)  
plot(cv.out)

# Two choices for lambda:
# - lambda.min: the value of lambda that achieves the minimum cvm (left dashed vertical line)
# - lambda.1se: the largest value of lambda (i.e., the largest regularization, the smallest df) 
#   whose cvm is within one-standard-error of the cvm of lambda.min (right dashed vertical line)

# Check how lambda.min and lambda.1se are computed
names(cv.out)

# Find lambda.min
cv.out$lambda.min
cv.out$lambda[which.min(cv.out$cvm)]

# Find lambda.1se
cv.out$lambda.1se
tmp.id = which.min(cv.out$cvm)
max(cv.out$lambda[cv.out$cvm < cv.out$cvm[tmp.id] + cv.out$cvsd[tmp.id]])

# ============================================================================
# RIDGE REGRESSION - PREDICTION
# ============================================================================

# Apply the two ridge regression models for prediction
myridge = glmnet(X[-test.id,], Y[-test.id], alpha=0, lambda = lam.seq)
Ytest.pred = predict(myridge, s = cv.out$lambda.1se, newx=X[test.id,])
mean((Ytest.pred - Y[test.id])^2)

Ytest.pred=predict(myridge, s = cv.out$lambda.min, newx=X[test.id,])
mean((Ytest.pred - Y[test.id])^2)

# ============================================================================
# LASSO REGRESSION
# ============================================================================

# Fit lasso regression models using glmnet
mylasso = glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
summary(mylasso)

# ============================================================================
# LASSO - PATH PLOTS AND CROSS VALIDATION
# ============================================================================

# Path plots
par(mfrow = c(1, 2))
plot(mylasso, label=TRUE, xvar = "norm")
plot(mylasso, label=TRUE, xvar = "lambda")
par(mfrow=c(1,1))

# Check degrees of freedom
mylasso$df

# Cross-validation for lasso
cv.out = cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
plot(cv.out)

# Try our own lambda sequences
# lam.seq =  exp(seq(-4, 2, length=100))
# lam.seq =  exp(seq(-6, -1, length=100))
# cv.out = cv.glmnet(X[-test.id,], Y[-test.id], alpha = 1, lambda = l am.seq)
# plot(cv.out)

# Check how lambda.min and lambda.1se are computed
cv.out$lambda.min
tmp.id=which.min(cv.out$cvm)
cv.out$lambda[tmp.id]

cv.out$lambda.1se
max(cv.out$lambda[cv.out$cvm < cv.out$cvm[tmp.id] + cv.out$cvsd[tmp.id]])

# ============================================================================
# LASSO - COEFFICIENTS
# ============================================================================

# How to retrieve Lasso coefficients?
mylasso.coef.min = predict(mylasso, s=cv.out$lambda.min, type="coefficients")
mylasso.coef.1se = predict(mylasso, s=cv.out$lambda.1se, type="coefficients")
cbind(mylasso.coef.min, mylasso.coef.1se)

# number of variables selected (including the intercept)
sum(mylasso.coef.1se != 0)

# names of selected non-intercept variables
row.names(mylasso.coef.1se)[which(mylasso.coef.1se != 0)[-1]]

# ============================================================================
# LASSO - PREDICTION
# ============================================================================

# Apply the two fitted models for prediction on test data
mylasso = glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
Ytest.pred = predict(mylasso, s = cv.out$lambda.min, newx = X[test.id,])
mean((Ytest.pred - Y[test.id])^2)

Ytest.pred = predict(mylasso, s = cv.out$lambda.1se, newx = X[test.id,])
mean((Ytest.pred - Y[test.id])^2)

# ============================================================================
# LASSO - REFIT TO REDUCE BIAS
# ============================================================================

# We refit an ordinary linear regression model using the variables
# selected by lambda.1se and then use it for prediction
mylasso.coef.1se = predict(mylasso, s = cv.out$lambda.1se, type="coefficients")
var.sel = row.names(mylasso.coef.1se)[which(mylasso.coef.1se != 0)[-1]]
var.sel; 

tmp.X = X[, colnames(X) %in% var.sel]
mylasso.refit = coef(lm(Y[-test.id] ~ tmp.X[-test.id, ]))
Ytest.pred = mylasso.refit[1] + tmp.X[test.id,] %*% mylasso.refit[-1]
mean((Ytest.pred - Y[test.id])^2) 