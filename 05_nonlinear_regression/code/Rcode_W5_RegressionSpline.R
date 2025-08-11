# Regression Splines

# --- Load Libraries ---
library(splines)
library(ggplot2)
# help(bs)
# help(ns)

# --- Spline Basis Functions ---
# Five knots, degree of freedom = 9
x = (1:199)/100
n = length(x)
m = 5
myknots = 2*(1:m)/(m+1)
myknots
# [1] 0.3333333 0.6666667 1.0000000 1.3333333 1.6666667

# The first four basis functions are global cubic polynomials: intercept, x, x^2, x^3
# The next five are truncated cubic functions at each knot
X = cbind(1, x, x^2, x^3)
for(i in 1:m){
    tmp = (x-myknots[i])^3
    tmp[tmp<0] = 0
    X = cbind(X, tmp)
}
plot(c(0,2), range(X), type="n", xlab="", ylab="")
title("Truncated Power Basis")
for(i in 1:(m+4)){
    tmp = X[,i]
    if (i<=4) mylty=1 else mylty=2
    lines(x[tmp!=0], tmp[tmp!=0], col=i, lty=mylty, lwd=2)
}
for(i in 1:m){
    points(myknots[i], 0, pty="m", pch=19, cex=2)
}

# --- B-spline basis using bs() ---
F = bs(x, knots = myknots, intercept = TRUE)
dim(F)
# [1] 199   9
mydf = m+4
# Visualize the basis functions
# (ggplot2 visualization)
tmpdata = data.frame(t = rep(1:n, mydf),
                     basisfunc=as.vector(F), 
                     type=as.factor(rep(1:mydf, each=n)))
ggplot(tmpdata, aes(x=t, y=basisfunc, color=type)) + 
  geom_path()

# If intercept=FALSE (default), bs() returns 8 columns
F = bs(x, knots = myknots)
dim(F)
mydf = m+3
tmpdata = data.frame(t = rep(1:n, mydf),
                     basisfunc=as.vector(F), 
                     type=as.factor(rep(1:mydf, each=n)))
ggplot(tmpdata, aes(x=t, y=basisfunc, color=type)) +
  geom_path()

# --- Natural Cubic Spline (NCS) basis using ns() ---
F = ns(x, knots=myknots, Boundary.knots=c(0,2), intercept=TRUE)
mydf = 7
tmpdata = data.frame(t = rep(1:n, mydf),
                     basisfunc=as.vector(F), 
                     type=as.factor(rep(1:mydf, each=n)))
ggplot(tmpdata, aes(x=t, y=basisfunc, color=type)) +
  geom_path()

# --- Regression Splines: The Birthrate Data ---
# The Birthrate Data: births per 2,000 people, with historical context
url = "https://liangfgithub.github.io/Data/birthrates.csv"
birthrates = read.csv(url)
names(birthrates) = c("year", "rate")
ggplot(birthrates, aes(x=year, y=rate)) + 
  geom_point() + geom_smooth(method="lm", se=FALSE)
# 'geom_smooth()' using formula 'y ~ x'

# --- How does R Count DF ---
# bs() and ns() can take either knots or df
# For cubic splines, df = m + 4 (m = number of knots)
# If you specify df, R places knots at quantiles
# The df argument in bs()/ns() is the number of columns in the design matrix (if intercept=FALSE, df = actual df - 1)

# Example: two knots at quantiles (df = 6, but returns 5 columns by default)
fit1 = lm(rate~bs(year, knots=quantile(year, c(1/3, 2/3))), data=birthrates)
fit2 = lm(rate~bs(year, df=5), data=birthrates)
fit3 = lm(rate~bs(year, df=6, intercept=TRUE), data=birthrates)
fit4 = lm(rate~bs(year, df=5, intercept=TRUE), data=birthrates)

plot(birthrates$year, birthrates$rate, ylim=c(90,280))
lines(spline(birthrates$year, predict(fit1)), col="red", lty=1)
lines(spline(birthrates$year, predict(fit2)), col="blue", lty=2)
lines(spline(birthrates$year, predict(fit3)), col="green", lty=3)
lines(spline(birthrates$year, predict(fit4)), lty=2, lwd=2)

# Predict on a fine grid
plot(birthrates$year, birthrates$rate, ylim=c(90,280))
year.grid = seq(from=min(birthrates$year), to=max(birthrates$year), length=200)
ypred = predict(fit1, data.frame(year=year.grid))
lines(year.grid, ypred, col="blue", lwd=2)

# --- Natural Cubic Spline Regression ---
fit1 = lm(rate~ns(year, knots=quantile(year, (1:4)/5)), data=birthrates)
fit2 = lm(rate~ns(year, df=5), data=birthrates)
fit3 = lm(rate~ns(year, df=6, intercept=TRUE), data=birthrates)

plot(birthrates$year, birthrates$rate, ylim=c(90,280))
lines(spline(birthrates$year, predict(fit1)), col="red", lty=1)
lines(spline(birthrates$year, predict(fit2)), col="blue", lty=2)
lines(spline(birthrates$year, predict(fit3)), col="green", lty=3)

# --- Prediction outside the data range ---
new = data.frame(year=1905:2015)
fit1 = lm(rate~bs(year, df=7), data=birthrates)
pred1 = predict(fit1, new)
# Warning: some 'x' values beyond boundary knots may cause ill-conditioned bases
fit2 = lm(rate~ns(year, df=7), data=birthrates)
pred2 = predict(fit2, new)
plot(birthrates$year, birthrates$rate, xlim=c(1905,2015),
     ylim=c(min(pred1,pred2), max(pred1,pred2)), 
     ylab="Birth Rate", xlab="Year") 
lines(new$year, pred1, col="red")
lines(new$year, pred2, col="blue")
legend("bottomleft", lty=rep(1,2),  col=c("red",  "blue" ), legend=c("CS with df=7", "NCS with df=7"))

# --- Select DF: Try cubic splines with different degree-of-freedoms ---
plot(birthrates$year, birthrates$rate, ylim=c(90,280))
lines(spline(birthrates$year, predict(lm(rate~bs(year, df=7), data=birthrates))), col="blue")
lines(spline(birthrates$year, predict(lm(rate~bs(year, df=14), data=birthrates))), col="red")
lines(spline(birthrates$year, predict(lm(rate~bs(year, df=19), data=birthrates))), col="black")
legend("topright", lty=rep(1,3), col=c("blue", "red", "black"), legend=c("df=8", "df=15", "df=20"))

# --- Model selection: 10-fold Cross-Validation for optimal df ---
set.seed(1234)
K = 10
n = nrow(birthrates)
fold.size = c(rep(9, 7), rep(8, 3))
fold.id = rep(1:K, fold.size)
fold.id
fold.id = fold.id[sample(1:n, n)]
fold.id
mydf = 10:30
mycv = rep(0, length(mydf))
for(i in 1:length(mydf)){
  m = mydf[i]-4  
  for(k in 1:K){
    id = which(fold.id == k)
    myknots = quantile(birthrates$year[-id], (1:m)/(m+1))
    myfit = lm(rate ~ bs(year, knots=myknots),
               data=birthrates[-id,])
    ypred = predict(myfit, newdata=birthrates[id,])
    mycv[i]=mycv[i] + sum((birthrates$rate[id] - ypred)^2)
  }
}
plot(mydf, mycv)

# End of Regression Splines R code
# Source: https://liangfgithub.github.io/Rcode_W5_RegressionSpline.html 