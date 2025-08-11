import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# copy from ISLP package
def summarize(results, conf_int=False):
    """
       https://github.com/intro-stat-learning/ISLP/blob/main/ISLP/models/__init__.py
    """
    tab = results.summary().tables[1]
    results_table = pd.read_html(tab.as_html(),
                                 index_col=0,
                                 header=0)[0]
    if not conf_int:
        columns = ['coef', 'std err', 't', 'P>|t|']
        return results_table[results_table.columns[:-2]]
    return results_table

# converted from R's poly()
# ref: https://stackoverflow.com/questions/39031172/how-poly-generates-orthogonal-polynomials-how-to-understand-the-coefs-ret
# compute orthogonal polynomials
def poly(x, degree):
    x_mean = x.mean(axis=0)
    x = x - x_mean
    Z = np.power.outer(x, np.arange(0, degree+1))

    # orthogonalize
    x = x.reshape(-1, 1)
    qr = np.linalg.qr(Z, mode='complete')
    z = np.zeros_like(Z)
    np.fill_diagonal(z, np.diag(qr[1]))
    Z = qr[0] @ z
    norm2 = (Z ** 2.0).sum(axis=0)
    alpha = (x * Z ** 2.0).sum(axis=0) / norm2 + x_mean
    Z = Z / np.sqrt(norm2)
    norm2 = np.insert(norm2, 0, 1.0, axis=0)

    return Z[:, 1:], alpha[:-1], norm2


def poly_predict(x_new, degree, alpha, norm2):
    n = x_new.shape[0]
    Z = np.ones((n, degree + 1))
    Z[:, 1] = x_new - alpha[0]
    for i in range(1, degree):
        Z[:, i + 1] = (x_new - alpha[i]) * Z[:, i]
        Z[:, i + 1] -= (norm2[i + 1] / norm2[i]) * Z[:, i - 1]

    Z = Z / np.sqrt(norm2[1:])

    return Z[:, 1:]

# Load the Wage data (from ISLR, Chapter 7), which contains income and demographic information 
# for males who reside in the central Atlantic region of the United States. 
# We will fit wage (in thousands of dollars) as a function of age.

url = "https://liangfgithub.github.io/Data/Wage.csv"
Wage = pd.read_csv(url)
Wage.describe()

age = Wage['age'].to_numpy()
wage = Wage['wage']

# Fit a Polynomial Regression Model
# Two Methods to specify the design matrix for polynomial regression:
# Method 1: Manually specify each term of the polynomial. For example, a cubic model would include terms like x, x^2, x^3.
# Method 2: Using the poly function. This is a more compact and efficient way, especially when d is large. 
# For example, poly(x,3) directly creates a cubic polynomial model.

# When you use the poly function, like poly(age, 3), you're not directly getting age, age^2, age^3. 
# Instead, you get three orthogonalized (and standardized) polynomial variables.

# For example, poly(age, 3) returns an n-by-3 matrix:
# each column has mean zero and sample variance 1, and they are orthogonal to each other.
# The 1st column is a linear combination of "age" and the intercept, capturing the linear component of the trend.
# The 2nd column is a linear combination of "age^2", "age", and the intercept, capturing the quadratic trend.
# The 3rd column is a linear combination of "age^3", age^2", "age", and the intercept, encapsulating the cubic trend.

# Method 1: Manual polynomial terms
X1 = np.power.outer(age, np.arange(1, 4))
M1 = sm.OLS(wage, sm.add_constant(X1)).fit()
summarize(M1)

# Method 2: Using poly function
X2, alpha, norm2 = poly(age, 3)
M2 = sm.OLS(wage, sm.add_constant(X2)).fit()
summarize(M2)

# Check properties of orthogonal polynomials
print(X2.shape)
print(X2.mean(axis=0))
print(np.round(X2.T @ X2, 4))

# Prediction
# Method 1 and Method 2 would produce different coefficients, but the fitted values (predictions) remain the same.

# Method 1 coefficients
fit1 = LinearRegression().fit(X1, wage)
fit1.intercept_, fit1.coef_

# Method 2 coefficients  
fit2 = LinearRegression().fit(X2, wage)
fit2.intercept_, fit2.coef_

# Prediction using Method 2
age_new = np.array([82])
age_new = poly_predict(age_new, 3, alpha, norm2)
fit2.predict(age_new)

# Prediction using Method 1
tmp = np.power.outer(82, np.arange(1, 4))
fit1.intercept_ + tmp @ fit1.coef_

# Note that although the coefficients in fit1 and fit2 are different, 
# the t-value and p-value for the last predictor remain identical. Why is this the case?

# In the context of linear regression, the t-test for a variable gauges its conditional contribution. 
# In both fit1 and fit2, the sole unique contribution of the cubic term of age comes only from the last variable, 
# irrespective of whether it's represented as "age^3" or "poly(age, 3)". 
# Thus, the p-value for the last variable in both models, which measures the unique contribution of the cubic term of age, 
# isolated from the influences of other variables, stays the same.

# Thus, if your primary concern is gauging the significance of the coefficient linked with the highest order term, 
# the choice of design matrix becomes inconsequential. Whether you utilize the direct design matrix or the orthogonal 
# polynomial design matrix, the significance of the highest order term remains consistent.

# This insight becomes particularly valuable when employing forward and backward selection methods to determine the optimal d.

# Fitted Curve
# fit1 and fit2 should return an identical fitted curve.

age_grid = np.arange(age.min(), age.max() + 1)
age_grid_poly = poly_predict(age_grid, 3, alpha, norm2)
y_grid = fit2.predict(age_grid_poly)

plt.plot(age_grid, y_grid)
plt.scatter(age, wage, marker='.', alpha=0.5, c='orange')
plt.title("Degree-3 Polynomial")
plt.xlabel("Age")
plt.ylabel("Wage")
plt.show() 