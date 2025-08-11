# R code

# --- Full model fit ---
full.model = lm( Y ~ ., data = myData)
summary(full.model)

# --- Subset selection using leaps ---
# install.packages("leaps")
library(leaps)
b = regsubsets(Y ~ ., data=myData, nvmax = p)
rs = summary(b)
rs$which

# --- Model selection criteria and plots ---
msize = 1:p
par(mfrow=c(1,2))
Aic = n*log(rs$rss/n) + 2*msize
Bic = n*log(rs$rss/n) + msize*log(n)
plot(msize, Aic, xlab="No. of Parameters", ylab = "AIC")
plot(msize, Bic, xlab="No. of Parameters", ylab = "BIC")

# --- Compare models selected by AIC and BIC ---
cbind(rs$which[which.min(Aic),], rs$which[which.min(Bic), ])

# --- Compare leaps BIC to calculated BIC ---
cbind(rs$bic, Bic, rs$bic - Bic)

# --- Find 2nd and 3rd best models (nbest = 3) ---
# ?regsubsets
b = regsubsets(Y ~ ., data=myData, nbest = 3, nvmax = p)
rs = summary(b)
rs$which
msize = apply(rs$which, 1, sum) - 1

# --- Model selection criteria and plots for nbest ---
par(mfrow=c(1,2))
Aic = n*log(rs$rss/n) + 2*msize
Bic = n*log(rs$rss/n) + msize*log(n)
plot(msize, Aic, xlab="No. of Parameters", ylab = "AIC")
plot(msize, Bic, xlab="No. of Parameters", ylab = "BIC")

# --- Plots for models with more than 2 parameters ---
plot(msize[msize > 2], Aic[msize > 2], ylim = c(-67, -62), xlab="No. of Parameters", ylab = "AIC")
plot(msize[msize > 2], Bic[msize > 2], ylim = c(-50, -45), xlab="No. of Parameters", ylab = "BIC") 