# Regression Splines Example
# Extracted and adapted from the Jupyter HTML notebook
# Includes B-spline (bs) and natural spline (ns) basis functions, and example usage

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splev, interp1d
from sklearn.linear_model import LinearRegression

# B-spline basis function (adapted from R's bs())
def bs(x, df=None, knots=None, boundary_knots=None, degree=3, include_intercept=False):
    """
    Generate B-spline basis matrix for input x.
    Parameters:
        x: array-like, input values
        df: degrees of freedom
        knots: internal knots
        boundary_knots: boundary knots
        degree: degree of the spline (default cubic)
        include_intercept: whether to include the intercept column
    Returns:
        basis: B-spline basis matrix
    """
    ord = 1 + degree
    if boundary_knots is None:
        boundary_knots = [np.min(x), np.max(x)]
    else:
        boundary_knots = np.sort(boundary_knots).tolist()
    oleft = x < boundary_knots[0]
    oright = x > boundary_knots[1]
    outside = oleft | oright
    inside = ~outside
    if df is not None:
        nIknots = df - ord + (1 - include_intercept)
        if nIknots < 0:
            nIknots = 0
        if nIknots > 0:
            knots = np.linspace(0, 1, num=nIknots + 2)[1:-1]
            knots = np.quantile(x[~outside], knots)
    Aknots = np.sort(np.concatenate((boundary_knots * ord, knots)))
    n_bases = len(Aknots) - (degree + 1)
    if any(outside):
        # Extrapolation handling
        scalef = sp.special.gamma(np.arange(1, ord + 1))[:, None]  # factorials
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        e = 1 / 4
        if any(oleft):
            k_pivot = (1 - e) * boundary_knots[0] + e * Aknots[ord]
            xl = np.power.outer(x[oleft] - k_pivot, np.arange(1, degree + 1))
            xl = np.c_[np.ones(xl.shape[0]), xl]
            tt = np.empty((xl.shape[1], n_bases), dtype=float)
            for j in range(xl.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)
            basis[oleft, :] = xl @ (tt / scalef)
        if any(oright):
            k_pivot = (1 - e) * boundary_knots[1] + e * Aknots[len(Aknots) - ord - 1]
            xr = np.power.outer(x[oright] - k_pivot, np.arange(1, degree + 1))
            xr = np.c_[np.ones(xr.shape[0]), xr]
            tt = np.empty((xr.shape[1], n_bases), dtype=float)
            for j in range(xr.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)
            basis[oright, :] = xr @ (tt / scalef)
        if any(inside):
            xi = x[inside]
            tt = np.empty((len(xi), n_bases), dtype=float)
            for i in range(n_bases):
                coefs = np.zeros((n_bases,))
                coefs[i] = 1
                tt[:, i] = splev(xi, (Aknots, coefs, degree))
            basis[inside, :] = tt
    else:
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (Aknots, coefs, degree))
    if include_intercept is False:
        basis = basis[:, 1:]
    return basis

# Natural spline basis function (adapted from R's ns())
def ns(x, df=None, knots=None, boundary_knots=None, include_intercept=False):
    """
    Generate natural spline basis matrix for input x.
    Parameters:
        x: array-like, input values
        df: degrees of freedom
        knots: internal knots
        boundary_knots: boundary knots
        include_intercept: whether to include the intercept column
    Returns:
        basis: natural spline basis matrix
    """
    degree = 3
    if boundary_knots is None:
        boundary_knots = [np.min(x), np.max(x)]
    else:
        boundary_knots = np.sort(boundary_knots).tolist()
    oleft = x < boundary_knots[0]
    oright = x > boundary_knots[1]
    outside = oleft | oright
    inside = ~outside
    if df is not None:
        nIknots = df - 1 - include_intercept
        if nIknots < 0:
            nIknots = 0
        if nIknots > 0:
            knots = np.linspace(0, 1, num=nIknots + 2)[1:-1]
            knots = np.quantile(x[~outside], knots)
    Aknots = np.sort(np.concatenate((boundary_knots * 4, knots)))
    n_bases = len(Aknots) - (degree + 1)
    if any(outside):
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        e = 1 / 4
        if any(oleft):
            k_pivot = boundary_knots[0]
            xl = x[oleft] - k_pivot
            xl = np.c_[np.ones(xl.shape[0]), xl]
            tt = np.empty((xl.shape[1], n_bases), dtype=float)
            for j in range(xl.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)
            basis[oleft, :] = xl @ tt
        if any(oright):
            k_pivot = boundary_knots[1]
            xr = x[oright] - k_pivot
            xr = np.c_[np.ones(xr.shape[0]), xr]
            tt = np.empty((xr.shape[1], n_bases), dtype=float)
            for j in range(xr.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)
            basis[oright, :] = xr @ tt
        if any(inside):
            xi = x[inside]
            tt = np.empty((len(xi), n_bases), dtype=float)
            for i in range(n_bases):
                coefs = np.zeros((n_bases,))
                coefs[i] = 1
                tt[:, i] = splev(xi, (Aknots, coefs, degree))
            basis[inside, :] = tt
    else:
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (Aknots, coefs, degree))
    const = np.empty((2, n_bases), dtype=float)
    for i in range(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        const[:, i] = splev(boundary_knots, (Aknots, coefs, degree), der=2)
    if include_intercept is False:
        basis = basis[:, 1:]
        const = const[:, 1:]
    qr_const = np.linalg.qr(const.T, mode='complete')[0]
    basis = (qr_const.T @ basis.T).T[:, 2:]
    return basis

# Example: Construct and plot truncated power basis functions
x = np.arange(0.01, 2, 0.01)
n = len(x)
m = 5
myknots = 2 * np.arange(1, m + 1) / (m + 1)
# myknots: array([0.333..., 0.666..., 1.0, 1.333..., 1.666...])

# Construct truncated power basis
X = np.vstack(([1.0] * n, x, x ** 2, x ** 3))
for i in range(m):
    tmp = (x - myknots[i]) ** 3
    tmp[tmp < 0] = 0
    X = np.vstack((X, tmp))

# Plot truncated power basis functions
plt.figure()
plt.title("Truncated Power Basis")
for i in range(m + 4):
    tmp = X[i]
    mylty = '-' if i < 4 else '--'
    plt.plot(x[tmp != 0], tmp[tmp != 0], linestyle=mylty)
for i in range(m):
    plt.plot(myknots[i], 0, "ok")
plt.tight_layout()
plt.show()

# Example: B-spline basis matrix and plot
F = bs(x, knots=myknots, include_intercept=True)
print("B-spline basis matrix shape:", F.shape)
plt.figure()
plt.plot(F)
plt.xlabel("t")
plt.ylabel("Basis Function")
plt.legend(range(F.shape[1]), title="Type", bbox_to_anchor=(1.01, 0.5), loc="center left")
plt.tight_layout()
plt.show() 