# PSL: Ridge and Lasso (scikit-learn)
# This script demonstrates loading data, splitting train/test, and applying Ridge and Lasso regression with cross-validation in Python.

# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# --- Load Data ---
# The Prostate data can be found at: https://hastie.su.domains/ElemStatLearn/data.html
# The first 8 columns are predictors; column 9 is the outcome/response.
myData = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/prostate.data", sep='\t')
myData = myData.drop(columns=[myData.columns[0], myData.columns[10]])
myData.head()

# --- Data Shape ---
myData.shape

# --- Prepare X and Y ---
X = myData.iloc[:, :-1]
Y = myData.iloc[:, -1]
X.shape, len(Y)

# --- Train/Test Split ---
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# --- Full Model (Ordinary Linear Regression) ---
full_model = lm()
full_model.fit(X_train, Y_train)
# Averaged MSE on the test set
print('Full Model Test MSE:', mean_squared_error(Y_test, full_model.predict(X_test)))

# --- Ridge Regression: Coefficient Path ---
alphas = np.logspace(-3, 3, 100)
ridge = Ridge(normalize=True)
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, Y_train)
    coefs.append(ridge.coef_)
coefs = np.array(coefs)
print('Ridge coefs shape:', np.shape(coefs))

# --- Ridge Regression: Plot Coefficient Path ---
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# --- Ridge Regression: Tune alpha by 10-fold CV with MSE ---
ridgecv = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(X_train, Y_train)
print('Best alpha (RidgeCV):', ridgecv.alpha_)

# --- Ridge Regression: Evaluate prediction performance ---
ridge_model = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge_model.fit(X_train, Y_train)
print('Ridge Test MSE:', mean_squared_error(Y_test, ridge_model.predict(X_test)))

# --- Ridge Regression: Coefficient values ---
print('Ridge coefficients:', pd.Series(ridge_model.coef_, index=X.columns))

# --- Ridge Regression: Manual CV Path Plot ---
mean_mse = []
std_mse = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha, normalize=True)
    mse_scores = -cross_val_score(ridge, X_train, Y_train, cv=10, scoring='neg_mean_squared_error')
    mean_mse.append(np.mean(mse_scores))
    std_mse.append(np.std(mse_scores) / np.sqrt(10))
mean_mse = np.array(mean_mse)
std_mse = np.array(std_mse)
min_idx = np.argmin(mean_mse)
alpha_min = alphas[min_idx]
threshold = mean_mse[min_idx] + std_mse[min_idx]
one_se_rule_idx = np.where(mean_mse <= threshold)[0][-1]
alpha_1se = alphas[one_se_rule_idx]
print('Ridge alpha_min:', alpha_min, 'alpha_1se:', alpha_1se)
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, mean_mse, label='Mean MSE', linewidth=2)
plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2)
plt.axvline(alpha_min, linestyle='--', color='r', label=f'Best alpha: {alpha_min:.2e}')
plt.axvline(alpha_1se, linestyle='--', color='g', label=f'alpha (1 SE): {alpha_1se:.2e}')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Ridge Cross-Validation Path')
plt.show()

# --- Lasso Regression: Coefficient Path ---
alphas = np.logspace(-4, -1, 100)
lasso = Lasso(normalize=True)
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, Y_train)
    coefs.append(lasso.coef_)
coefs = np.array(coefs)
print('Lasso coefs shape:', np.shape(coefs))

# --- Lasso Regression: Plot Coefficient Path ---
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# --- Lasso Regression: Tune alpha by 10-fold CV with MSE ---
lassocv = LassoCV(alphas=alphas, cv=10, normalize=True)
lassocv.fit(X_train, Y_train)
print('Best alpha (LassoCV):', lassocv.alpha_)

# --- Lasso Regression: Evaluate prediction performance ---
lasso_model = Lasso(alpha=lassocv.alpha_, normalize=True)
lasso_model.fit(X_train, Y_train)
print('Lasso Test MSE:', mean_squared_error(Y_test, lasso_model.predict(X_test)))

# --- Lasso Regression: Coefficient values ---
print('Lasso coefficients:', pd.Series(lasso_model.coef_, index=X.columns))

# --- Lasso Regression: Variables not selected ---
print('Lasso zero coefficients:', X.columns[np.where(lasso_model.coef_ == 0)])

# --- Lasso Regression: Manual CV Path Plot ---
mean_mse = np.mean(lassocv.mse_path_, axis=1)
std_mse = np.std(lassocv.mse_path_, axis=1) / np.sqrt(10)
cv_alphas = lassocv.alphas_
min_idx = np.argmin(mean_mse)
alpha_min = cv_alphas[min_idx]
threshold = mean_mse[min_idx] + std_mse[min_idx]
alpha_1se = max(cv_alphas[np.where(mean_mse <= threshold)])
print('Lasso alpha_min:', alpha_min, 'alpha_1se:', alpha_1se)
plt.figure(figsize=(10, 6))
plt.semilogx(cv_alphas, mean_mse, label='Mean MSE', linewidth=2)
plt.fill_between(cv_alphas, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2)
plt.axvline(alpha_min, linestyle='--', color='r', label=f'Best alpha: {alpha_min:.2e}')
plt.axvline(alpha_1se, linestyle='--', color='g', label=f'alpha (1 SE): {alpha_1se:.2e}')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Cross-Validation Path')
plt.show()

# --- Lasso Regression: Evaluate prediction performance with alpha_1se ---
lasso_1se_model = Lasso(alpha=alpha_1se, normalize=True)
lasso_1se_model.fit(X_train, Y_train)
print('Lasso 1SE Test MSE:', mean_squared_error(Y_test, lasso_1se_model.predict(X_test)))

# --- Lasso Regression: Refit to Reduce Bias ---
nonzero_indices = np.where(lasso_1se_model.coef_ != 0)[0]
lm_refit = lm()
lm_refit.fit(X_train.iloc[:, nonzero_indices], Y_train)
print('Refit Test MSE:', mean_squared_error(Y_test, lm_refit.predict(X_test.iloc[:, nonzero_indices]))) 