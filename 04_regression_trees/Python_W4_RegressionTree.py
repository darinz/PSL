# PSL: Regression Tree - Python Implementation
# This script demonstrates fitting and pruning regression trees using scikit-learn on the Boston Housing dataset.
# It includes data loading, model fitting, visualization, cost complexity pruning, and cross-validation for optimal alpha selection.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

# Load the Boston Housing dataset
url = "https://liangfgithub.github.io/Data/HousingData.csv"
Housing = pd.read_csv(url)
print(Housing.head())

# Prepare features and target
X = Housing.drop("Y", axis=1)
y = Housing["Y"]

# Fit a regression tree with a maximum of 10 leaf nodes
tr1 = DecisionTreeRegressor(max_leaf_nodes=10)
tr1.fit(X, y)

# Visualize the fitted regression tree
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(tr1, feature_names=X.columns, filled=True, impurity=False, ax=ax, fontsize=10)
fig.tight_layout()
plt.show()

# ---
# Complexity Pruning: Cost Complexity Pruning Path
# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
# The cost complexity criterion is: (1/n) * RSS(T) + alpha * |T|

# Get the cost complexity pruning path
path = tr1.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print("Alpha and impurity path:")
print(np.column_stack((ccp_alphas, impurities)))

# The difference between adjacent impurities is the corresponding alpha value
print("Difference between impurities:")
print(np.diff(impurities))

# Plot total impurity vs effective alpha
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

# ---
# Cross-validation to find the best alpha (complexity parameter)
# Compute the geometric mean of adjacent alphas to get beta sequence
ccp_betas = np.sqrt(ccp_alphas[1:] * ccp_alphas[:-1])
ccp_betas = np.append(ccp_betas, 100)
n_beta = len(ccp_betas)
print("Beta sequence:")
print(ccp_betas)

# 10-fold cross-validation for each beta
folds = 10
cv = KFold(n_splits=folds, shuffle=True)
scores_mean = []
scores_se = []
for beta in ccp_betas:
    clf = DecisionTreeRegressor(ccp_alpha=beta)
    score = cross_val_score(clf, X, y, cv=cv)
    scores_mean.append(score.mean())
    scores_se.append(score.std())

# Flip for plotting (to match increasing complexity)
scores = 1.0 - np.flip(np.array(scores_mean))
err = np.flip(scores_se) / np.sqrt(folds)
score_1se = err[scores.argmin()] + scores.min()

# Plot cross-validation relative error
fig, ax = plt.subplots()
ax.errorbar(range(n_beta), y=scores, yerr=err, marker='o', mfc='none')
ax.axhline(score_1se, linestyle='--')
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Cross-validation Relative Error")
# Set xticks to show beta values
xticks = ax.get_xticks()
ticks = np.flip(ccp_betas)[xticks[1:-1].astype(int)]
ticks = np.round(ticks, 4)
ticks[0] = float('inf')
ax.set_xticks(xticks[1:-1])
ax.set_xticklabels(ticks)
fig.set_figwidth(10)
fig.set_figheight(5)
plt.show()

# ---
# If you’re uncertain whether the cross-validation error has reached its minimum, consider starting with a larger tree.
# Example: Fit a larger tree and repeat the pruning and cross-validation process
tr2 = DecisionTreeRegressor(max_leaf_nodes=40)
tr2.fit(X, y)
path = tr2.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_betas = np.sqrt(ccp_alphas[1:] * ccp_alphas[:-1])
ccp_betas = np.append(ccp_betas, 100)
n_beta = len(ccp_betas)

scores_mean = []
scores_se = []
for beta in ccp_betas:
    clf = DecisionTreeRegressor(ccp_alpha=beta)
    score = cross_val_score(clf, X, y, cv=cv)
    scores_mean.append(score.mean())
    scores_se.append(score.std())

scores = 1.0 - np.flip(np.array(scores_mean))
err = np.flip(scores_se) / np.sqrt(folds)
score_1se = err[scores.argmin()] + scores.min()

fig, ax = plt.subplots()
ax.errorbar(range(len(ccp_betas)), y=scores, yerr=err, marker='o', mfc='none')
ax.axhline(score_1se, linestyle='--')
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("X-val Relative Error")
xticks = ax.get_xticks()
ticks = np.flip(ccp_betas)[xticks[1:-1].astype(int)]
ticks = np.round(ticks, 4)
ticks[0] = float('inf')
ax.set_xticks(xticks[1:-1])
ax.set_xticklabels(ticks)
fig.set_figwidth(10)
fig.set_figheight(5)
plt.show()

# ---
# Choosing the optimal complexity parameter (CP) value
# Use either the minimum cross-validation error or the One Standard Error (1-SE) rule
new_betas = np.flip(ccp_betas)
print("Index of minimum error:", scores.argmin())
print("alpha.min:", new_betas[scores.argmin()])
print("alpha.1se:", np.max(new_betas[scores < score_1se])) 